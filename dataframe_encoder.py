# Methods for encoding a dataframe as produced from the Perovskite Database
from functools import partial

import numpy as np
from sklearn.preprocessing import TargetEncoder
import pandas as pd


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


def filter_compositions(df, filter_dict):
    df_filtered = df.copy()
    df_filtered[list(filter_dict.keys())] = df_filtered[list(filter_dict.keys())].map(
        trim_ions_string
    )
    def check_condition(ion_list, ions_valid):
        result = True
        for ion in ion_list:
            if not ion in ions_valid:
                result = False
        return result

    for comp_col, ions_valid in filter_dict.items():
        mapfunc = partial(check_condition, ions_valid=ions_valid)
        df_filtered[comp_col] = df_filtered[comp_col].map(mapfunc)

    return df[df_filtered[list(filter_dict.keys())].apply(all, axis=1)]


def filter_singlelayer(df, cols_comp):
    mask = df[cols_comp].map(lambda x: not "|" in x)
    return df[mask.apply(all, axis=1)]


def filter_valid_ratio(df, cols_ratio):
    mask = df[cols_ratio].map(lambda x: not "x" in x)
    return df[mask.apply(all, axis=1)]


def filter_common(df, cols, thresh):
    index_common = {}
    for col in cols:
        common_values = []
        freq_sum = 0
        count = df[col].value_counts(normalize=True)
        for freq, value in zip(count.values, count.index):
            freq_sum += freq
            common_values.append(value)
            if freq_sum > thresh:
                break
        indices = set(df[df[col].map(lambda x: x in common_values)].index)
        if not index_common: # check if empty
            index_common = indices
        else:
            index_common = index_common & indices
    return df.loc[list(index_common)]


class CompositionEncoder:
    def __init__(self):
        self.ions_unique = []
        self.n_categories = None

    def fit(self, ions_stacks: pd.Series):
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

    def transform(self, ions_stacks: pd.Series, ions_ratios: pd.Series):
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
        df_composition = pd.DataFrame(
            np.vstack(compositions), columns=self.ions_unique,
            index=ions_stacks.index)
        return df_composition


class CompositionEncoder_DF:
    """Encode multiple composition columns in the same dataframe."""
    def __init__(self, composition_dict):
        self.composition_dict = composition_dict

    def fit(self, df):
        self.comp_encoders = {}
        self.ions = set()
        for ion_col, ratio_col in self.composition_dict.items():
            comp_enc = CompositionEncoder()
            comp_enc.fit(df[ion_col])
            if not set(comp_enc.ions_unique).isdisjoint(self.ions):
                raise ValueError(
                    f"Following ion names are present in multiple columns:\
                    {set(comp_enc.ions_unique) & self.ions}")
            else:
                self.ions.update(set(comp_enc.ions_unique))
            self.comp_encoders[ion_col] = comp_enc

    def transform(self, df, append=False):
        comp_cols = []
        for ion_col, ratio_col in self.composition_dict.items():
            comp_cols.append(self.comp_encoders[ion_col].transform(
                df[ion_col], df[ratio_col]))
        df_comp = pd.concat(comp_cols, axis=1)
        if append:
            return pd.concat([df, df_comp], axis=1)
        else:
            return df_comp

    def fit_transform(self, df, append=False):
        self.fit(df)
        return self.transform(df, append)


def aggregate_duplicate_rows(df, cols_aggregate, cols_mean, dropna=False):
    """Return a dataframe with aggregated duplicate rows.
    
    Rows that all have the same values for columns in cols_aggregate, are
    aggregated, such that for these columns keep unique combinations of values,
    and over the columns in cols_mean, the average or mean is taken, as well as
    the standard deviation. For all other columns in the dataframe, the first
    entry of the group is taken as the aggregate.
    
    NOTE: columns with integer dtypes and missing NaN values are changed to
    float by pandas. This could be prevented with dtype=pd.Int64Dtype() if need
    be in the future.

    Example: (also see test for this function)
    >>> df = pd.DataFrame({
        ... 'id':['a', 'b', 'c', 'd'],
        ... 'A': [1, 1, 2, 2],
        ... 'B': [1, 2, 2, 2],
        ... 'c': [1., 2., 3., 3.]})
    >>> aggregate_duplicate_rows(df, ['A', 'B'], ['c'])
        id  A   B   c   c_std
    0   a   1   1   1.  None
    1   b   1   2   2.  None
    2   c   2   2   3.  0.
    """
    cols_all = df.keys().to_list()
    cols_other = list(set(cols_all) - set(cols_mean))
    grouped = df.groupby(cols_aggregate, as_index=False, dropna=dropna)
    agg_dict = {
        **{col: ["mean", "std"] for col in cols_mean},
        **{col: "first" for col in cols_aggregate},
        **{col: "first" for col in cols_other}
    }
    df_agg = grouped.agg(agg_dict)
    col_names = ['_'.join(col_name).strip().replace("_mean","").replace("_first","") \
        for col_name in df_agg.columns.values]
    df_agg.columns = col_names
    return df_agg


def get_value_space(df, cols_type_dict, dropna=False):
    """Return the space of values for columns cols_type_dict in DataFrame df.
    
    Returned value is a list of lists and/or dicts, where each list describes
    the space for a feature coresponding to a column in cols.
    cols_type_dict is a dict with column names as keys and dtype as values.
    
    float columns are turned into minimum and maximum, integer columns are
    turned into minimum and maximum with stepsize 1, categorical or object
    columns are turned into a list of the observed values."""
    if dropna:
        df = df.dropna(subset=cols_type_dict.keys())
    value_spaces = []
    for col, dtype in cols_type_dict.items():
        counts = df.value_counts(subset=col)
        if dtype in [np.float64, "float"]:
            value_space = {"low": min(counts.index), "high": max(counts.index)}
            value_spaces.append(value_space)
        if dtype in [np.int64, "int"]:
            value_space = {
                "low": min(counts.index), "high": max(counts.index), "step": 1}
            value_spaces.append(value_space)
        elif dtype in [np.dtypes.ObjectDType, "str", "category"]:
            value_space = counts.index.to_list()
            value_spaces.append(sorted(value_space))
    return value_spaces


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
        # encode compositions
        self.ions_lists = {} # to collect the observed ion names
        self.ion_encoders = {} # to collect the composition encoders
        for comp_ion_col in self.cols_composition_dict.keys():
            enc = CompositionEncoder()
            enc.fit(df[comp_ion_col].to_list())
            self.ion_encoders[comp_ion_col] = enc # save the encoder in dict
            self.ions_lists[comp_ion_col] = [ion+"_a_site_coefficient" \
                for ion in enc.ions_unique]

        self.enc_target = TargetEncoder(
            smooth="auto", target_type="continuous")
        # TODO: add support for binary targets
        X = df[self.cols_category]
        y = df[self.target]
        self.enc_target.fit(X, y)

    def transform(self, df_in, append=False):
        """Append determines whether the transformed columns are appended to df
        (True) or whether a new DataFrame is created (False)."""
        if append:
            df_out = df_in.copy()
        else:
            df_out = pd.DataFrame({})
        for ion_name_col, ion_ratio_col in self.cols_composition_dict.items():
            comp = self.ion_encoders[ion_name_col].transform(
                df_in[ion_name_col], df_in[ion_ratio_col])
            col_names = list(comp.columns.values)
            col_names_dict = {name: ion_name_col+"_"+name for name in col_names}
            comp = comp.rename(columns=col_names_dict)
            df_out = pd.concat([df_out, comp], axis=1)

        X = df_in[self.cols_category]
        X_trans = self.enc_target.transform(X)
        for i, col_category in enumerate(self.cols_category):
            df_out.loc[:, col_category+"_out"] = X_trans[:, i]
        
        return df_out

    def fit_transform(self, df_in, append=False):
        """Fit the encoder using data in df, and return transformed datapoints.
        
        This uses the cross fitting of TargetEncoder, so overfitting the
        encoding is avoided."""
        if append:
            df_out = df_in.copy()
        else:
            df_out = pd.DataFrame({})
        # encode compositions
        self.ions_lists = {} # to collect the observed ion names
        self.ion_encoders = {} # to collect the composition encoders
        for comp_ion_col in self.cols_composition_dict.keys():
            enc = CompositionEncoder()
            enc.fit(df_in[comp_ion_col].to_list())
            self.ion_encoders[comp_ion_col] = enc # save the encoder in dict
            self.ions_lists[comp_ion_col] = [ion+"_a_site_coefficient" \
                for ion in enc.ions_unique]
        # transform compositions
        for ion_name_col, ion_ratio_col in self.cols_composition_dict.items():
            comp = self.ion_encoders[ion_name_col].transform(
                df_in[ion_name_col], df_in[ion_ratio_col])
            col_names = list(comp.columns.values)
            col_names_dict = {name: ion_name_col+"_"+name for name in col_names}
            comp = comp.rename(columns=col_names_dict)
            df_out = pd.concat([df_out, comp], axis=1)
        # encode categorical variables
        self.enc_target = TargetEncoder(
            smooth="auto", target_type="continuous")
        # TODO: add support for binary targets
        X = df_in[self.cols_category]
        y = df_in[self.target]
        X_trans = self.enc_target.fit_transform(X, y)
        for i, col_category in enumerate(self.cols_category):
            df_out.loc[:, col_category+"_out"] = X_trans[:, i]

        return df_out