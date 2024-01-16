# Methods for encoding a dataframe as produced from the Perovskite Database
import numpy as np
from sklearn.preprocessing import TargetEncoder


def _isvalid(number):
    if number is None or np.isnan(number):
        return False
    else:
        return True


def trim_ions_string(ions_string):
    # NOTE: | in the stack string denotes layering, we ignore this and split
    # on compositions (denoted by "; ") and layering agnostically
    return ions_string.replace("; ", " ").replace(" | ", " ").split()

def trim_ions_ratio(ions_ratio_string):
    # NOTE: some strings might have x as a ratio, we replace this as 1
    temp = ions_ratio_string
    ions_ratio_string = ions_ratio_string.replace("x", "1")
    coefficients = ions_ratio_string.replace("; ", " ").replace(" | ", " ").split()
    return [float(coefficient) for coefficient in coefficients]


class CompositionEncoder:
    def __init__(self):
        self.ions_unique = []
        self.n_categories = None

    def fit(self, ions_stacks):
        self.ions_unique = []
        for ions_stack in ions_stacks:
            if not isinstance(ions_stack, str):
                continue
            ions_stack = trim_ions_string(ions_stack)
            for ion in ions_stack:
                if not ion in self.ions_unique:
                    self.ions_unique.append(ion)
        self.n_categories = len(self.ions_unique)
        if len(self.ions_unique) == 0:
            raise ValueError("No valid ion names found in column.")

    def transform(self, ions_stacks, ions_ratios):
        compositions = []
        for ions_stack, ions_ratio in zip(ions_stacks, ions_ratios):
            if not isinstance(ions_stack, str) or not isinstance(ions_ratio, str):
                # if there are no ion names or ratios, append a nan vector
                composition = np.zeros(self.n_categories)*np.nan
                compositions.append(composition)
                continue
            ions_stack = trim_ions_string(ions_stack)
            ions_ratio = trim_ions_ratio(ions_ratio)
            composition = np.zeros(self.n_categories)
            indices = [self.ions_unique.index(ion) for ion in ions_stack]
            for index, ratio in zip(indices, ions_ratio):
                composition[index] += ratio
            composition /= np.sum(composition)
            compositions.append(composition)

        return np.vstack(compositions)


class DFEncoder:
    def __init__(self,
        target,
        cols_category,
        cols_composition_dict
    ):
        self.target = target
        self.cols_category = cols_category
        self.cols_composition_dict = cols_composition_dict
        self.features = []
        self.agg_dict = {
            target: ["mean", "std"],
            **{col: "first" for col in cols_category},
            **{col: "first" for col in cols_composition_dict.keys()},
            **{col: "first" for col in cols_composition_dict.values()}}
        self.cols_in = self.cols_category \
            + list(self.cols_composition_dict.keys()) \
            + list(self.cols_composition_dict.values()) #+ self.cols_target
        self.ions_list = None
    
    def fit(self, df):
        """Fit the encoder to the data in df using the columns with which the
        encoder was initialized. NOTE: using fit and transform on this encoder
        does not make use of cross fitting from TargetEncoder.fit_transform
        which can lead to overfitting."""
        grouped = df.groupby(self.cols_in, as_index=False, dropna=False)
        df_fit = grouped.agg(self.agg_dict)
        col_names = ['_'.join(col_name).strip().replace("_mean","").replace("_first","") \
            for col_name in df_fit.columns.values]
        df_fit.columns = col_names
        # drop rows that have no target value, even after aggregation
        df_fit = df_fit.dropna(subset=self.target)

        # encode compositions
        self.ions_lists = {} # to collect the observed ion names
        self.ion_encoders = {} # to collect the composition encoders
        for comp_ion_col in self.cols_composition_dict.keys():
            enc = CompositionEncoder()
            enc.fit(df_fit[comp_ion_col].to_list())
            self.ion_encoders[comp_ion_col] = enc # save the encoder in dict
            self.ions_lists[comp_ion_col] = [ion+"_a_site_coefficient" \
                for ion in enc.ions_unique]

        self.enc_target = TargetEncoder(
            smooth="auto", target_type="continuous")
        # TODO: add support for binary targets
        X = df_fit[self.cols_category]
        y = df_fit[self.target]
        self.enc_target.fit(X, y)

    def transform(self, df):
        grouped = df.groupby(self.cols_in, as_index=False, dropna=False)
        df_fit = grouped.agg(self.agg_dict)
        col_names = ['_'.join(col_name).strip().replace("_mean","").replace("_first","") \
            for col_name in df_fit.columns.values]
        df_fit.columns = col_names
        # drop rows that have no target value, even after aggregation
        df_fit = df_fit.dropna(subset=self.target)

        comps = []
        for ion_name_col, ion_ratio_col in self.cols_composition_dict.items():
            comp = self.ion_encoders[ion_name_col].transform(
                df_fit[ion_name_col], df_fit[ion_ratio_col])
            comps.append(comp)
        comp_all = np.concatenate(comps, axis=1)

        X = df_fit[self.cols_category]
        X_trans = self.enc_target.transform(X)
        X_combined = np.concatenate((X_trans, comp_all), axis=1)

        return X_combined

    def fit_transform(self, df):
        """Fit the encoder using data in df, and return transformed datapoints.
        
        This uses the cross fitting of TargetEncoder, so overfitting the
        encoding is avoided."""
        grouped = df.groupby(self.cols_in, as_index=False, dropna=False)
        df_fit = grouped.agg(self.agg_dict)
        col_names = ['_'.join(col_name).strip().replace("_mean","").replace("_first","") \
            for col_name in df_fit.columns.values]
        df_fit.columns = col_names
        # drop rows that have no target value, even after aggregation
        df_fit = df_fit.dropna(subset=self.target)

        # encode compositions
        self.ions_lists = {} # to collect the observed ion names
        self.ion_encoders = {} # to collect the composition encoders
        for comp_ion_col in self.cols_composition_dict.keys():
            enc = CompositionEncoder()
            enc.fit(df_fit[comp_ion_col].to_list())
            self.ion_encoders[comp_ion_col] = enc # save the encoder in dict
            self.ions_lists[comp_ion_col] = [ion+"_a_site_coefficient" \
                for ion in enc.ions_unique]

        self.enc_target = TargetEncoder(
            smooth="auto", target_type="continuous")
        # TODO: add support for binary targets
        X = df_fit[self.cols_category]
        y = df_fit[self.target]
        X_trans = self.enc_target.fit_transform(X, y)

        comps = []
        for ion_name_col, ion_ratio_col in self.cols_composition_dict.items():
            comp = self.ion_encoders[ion_name_col].transform(
                df_fit[ion_name_col], df_fit[ion_ratio_col])
            comps.append(comp)
        comp_all = np.concatenate(comps, axis=1)

        X_combined = np.concatenate((X_trans, comp_all), axis=1)

        return X_combined