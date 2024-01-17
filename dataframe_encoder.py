# Methods for encoding a dataframe as produced from the Perovskite Database
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
        '''grouped = df_fit.groupby(self.cols_in, as_index=False, dropna=False)
        df_fit = grouped.agg(self.agg_dict)
        col_names = ['_'.join(col_name).strip().replace("_mean","").replace("_first","") \
            for col_name in df_fit.columns.values]
        df_fit.columns = col_names'''
        # drop rows that have no target value, even after aggregation
        #df_fit = df_fit.dropna(subset=self.target)

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
        # TODO (IMPORTANT): get rid of aggregation, this should only be
        # necesssary in fitting
        '''grouped = df_in.groupby(self.cols_in, as_index=False, dropna=False)
        df_out = grouped.agg(self.agg_dict)
        col_names = ['_'.join(col_name).strip().replace("_mean","").replace("_first","") \
            for col_name in df_out.columns.values]
        df_out.columns = col_names'''
        # drop rows that have no target value, even after aggregation
        #df_out = df_out.dropna(subset=self.target)
        for ion_name_col, ion_ratio_col in self.cols_composition_dict.items():
            comp = self.ion_encoders[ion_name_col].transform(
                df_in[ion_name_col], df_in[ion_ratio_col])
            for i, ion in enumerate(
                    self.ion_encoders[ion_name_col].ions_unique):
                df_out.loc[:, ion_name_col+"_"+ion] = comp[:, i]

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
            for i, ion in enumerate(
                    self.ion_encoders[ion_name_col].ions_unique):
                df_out.loc[:, ion_name_col+"_"+ion] = comp[:, i]
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