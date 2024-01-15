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
        cols_target,
        cols_category,
        cols_composition
    ):
        self.cols_target = cols_target
        self.cols_category = cols_category
        self.cols_composition = cols_composition
        self.features = []
        agg_dict = {
            **{col: ["mean", "std"] for col in cols_target},
            **{col: "first" for col in cols_category},
            **{col: "first" for col in cols_composition}}
        self.cols_in = self.cols_category + self.cols_composition
        self.ions_list = None
    
    def fit(self, df):
        grouped = df.groupby(self.cols_in, as_index=False)
        df_fit = grouped.agg(agg_dict)
        col_names = ['_'.join(col_name).strip().replace("_mean","").replace("_first","") \
            for col_name in df_fit.columns.values]
        df_fit.columns = col_names
        # drop rows that have no target value, even after aggregation
        df_fit = df_fit.dropna(subset=self.cols_target)

        # encode compositions
        self.ions_list = [] # to collect the observed ion names
        self.enc_comp_a = CompositionEncoder()
        self.enc_comp_a.fit(
            df_fit["Perovskite_composition_a_ions"].to_list())
        ions_list += [ion+"_a_site_coefficient" \
            for ion in self.enc_comp_a.ions_unique]

        self.enc_comp_b = CompositionEncoder()
        self.enc_comp_b.fit(
            df_fit["Perovskite_composition_b_ions"].to_list())
        self.ions_list += [ion+"_b_site_coefficient" \
            for ion in self.enc_comp_b.ions_unique]

        self.enc_comp_c = CompositionEncoder()
        self.enc_comp_c.fit(
            df_fit["Perovskite_composition_c_ions"].to_list())
        self.ions_list += [ion+"_c_site_coefficient" \
            for ion in self.enc_comp_c.ions_unique]

        self.enc_target = TargetEncoder(smooth="auto")
        X = df_fit[self.cols_category]
        y = df_fit[self.cols_target]
        enc.fit(X, y)

    def transform(self, df):
        comp_a = self.enc_comp_a.transform(
            df["Perovskite_composition_a_ions"],
            df["Perovskite_composition_a_ions_coefficients"])
        comp_b = self.enc_comp_b.transform(
            df["Perovskite_composition_b_ions"],
            df["Perovskite_composition_b_ions_coefficients"])
        comp_c = self.enc_comp_c.transform(
            df["Perovskite_composition_c_ions"],
            df["Perovskite_composition_c_ions_coefficients"])
        comp_all = np.concatenate((comp_a, comp_b, comp_c), axis=1)

        X = df[self.cols_in]
        y = df[self.cols_target]
        X_trans = enc.fit_transform(X, y)
        X_combined = np.concatenate((X_trans, comp_all), axis=1)

        return X_combined, y
