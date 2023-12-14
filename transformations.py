from typing import List

from ase.formula import Formula
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np


class StackEncoder:
    """Encodes the device stacks by splitting the stack string and
    one-hot-encoding individual layers."""
    def __init__(self, min_frequency=None):
        self.enc = OneHotEncoder(
            sparse_output=False, min_frequency=min_frequency)
        self.min_frequency = min_frequency

    def fit(self, stack_list: List[str]):
        stack_list = [trim_stack_string(stack) for stack in stack_list]
        flattened = np.concatenate(list(stack_list)).reshape(-1, 1)
        self.enc.fit(flattened)
        if self.min_frequency is not None:
            self.n_categories = (
                len(self.enc.categories_[0]) -
                len(self.enc.infrequent_categories_[0]) + 1)
        else:
            self.n_categories = len(self.enc.categories_[0])

    def transform(self, stack_list):
        """Transform list of device stacks to summed one hot encoding.
        
        stack_list = [
            ['SLG', 'ITO', 'PEDOT:PSS', 'Perovskite', 'PCBM-60', 'Au'],
            ['SLG', 'FTO', 'TiO2-c', 'TiO2-mp', 'Perovskite', 'Au'],
            ['SLG', 'ITO', 'SLG', 'Perovskite', 'Au']
        ] -> np.array([
            [1 1 1 1 1 1 0 0 0 0...],
            [1 0 0 1 0 1 1 1 1 0...],
            [2 1 0 1 0 1 0 0 0 0...]
        ])
        """
        n_categories = len(self.enc.categories_[0])
        stacks_tr_list = []
        for i, stack in enumerate(stack_list):
            stack = trim_stack_string(stack)
            if len(stack)==0:
                stacks_tr_list.append(np.zeros((n_categories)))
            else:
                stack = np.array(stack).reshape(-1, 1)
                stack_tr = self.enc.transform(stack)
                stack_tr = np.sum(stack_tr, axis=0)
                stacks_tr_list.append(stack_tr)
        return np.vstack(stacks_tr_list)

    def inverse_transform(self, stacks_tr: np.array):
        """Transform list of summed one hot encoded device stack back to list
        of device stacks. Note: stack layers will not be in the same order,
        as before the original transform due to the summation.
        
        For an example, see transform (but inverse).
        """
        stack_list = []
        for stack_tr in stacks_tr:
            stack = []
            for idx, count in enumerate(stack_tr):
                count = int(count)
                if count > 0:
                    one_hot = np.zeros(self.n_categories)
                    one_hot[idx] = 1
                    layer = self.enc.inverse_transform(one_hot.reshape(1, -1))
                    for _ in range(count):
                        stack.append(layer[0, 0])
            stack_list.append(stack)
        return stack_list




def trim_stack_string(stack: str):
    """Return list of strings of different layers in the device stack."""
    if not isinstance(stack, str):
        return []
    stack = stack.strip('"[]"')
    stack = stack.split(sep=", ")
    stack = [layer.strip("'") if isinstance(layer, str) else layer for layer in stack]
    #stack.remove("Perovskite")
    return stack


def get_compositions_vector_column(formula_series: pd.Series) -> pd.Series:
    """Return a np.array with composition vectors computed from df.
    
    df needs to have a column 'chemical_formula_hill'. """
    symbols_all = set()
    # get set of all elements in the dataset
    for formula_hill in formula_series:
        if not isinstance(formula_hill, str):
            continue
        formula_dict = Formula(formula_hill).count()
        symbols_all.update(formula_dict.keys())
    symbols_all_zero_dict = {symbol: 0 for symbol in symbols_all}

    compositions = []
    indices = []
    for index, formula_hill in enumerate(formula_series):
        indices.append(index)
        if not isinstance(formula_hill, str):
            compositions.append(np.zeros(len(symbols_all_zero_dict)))
            continue
        dict_copy = dict(symbols_all_zero_dict)
        formula_dict = Formula(formula_hill).count()
        n_atoms = sum(formula_dict.values())

        # normalize values in formula_dict
        formula_dict_normalized = {element: value/n_atoms for element, value in formula_dict.items()}
        for element, value in formula_dict_normalized.items():
            dict_copy[element] = value
        composition = np.array(list(dict_copy.values()))
        compositions.append(composition)
    
    return np.array(compositions)


def filter_uncommon(df: pd.DataFrame, prop_count_dict: dict) -> pd.DataFrame:
    """Filter out rows with uncommon categorical values.
    
    prop_count_dict defines which properties should be filtered, and how many
    unique values should be left at most."""
    common_value_dict = {}
    df_common = df.copy()
    for col, num_vals in prop_count_dict.items():
        common_values = df[col].value_counts()[:num_vals].keys()
        common_value_dict[col] = common_values
    #df_common_formula = df.loc[df['chemical_formula_descriptive'].isin(counts[:n].keys())]
    for col, values in common_value_dict.items():
        df_common = df_common.loc[df_common[col].isin(values)]
    return df_common


def int_encode_column(col: pd.Series, return_classes=False):
    """Encode categorical values from col using integers. Return encoded column."""
    le = LabelEncoder()
    le.fit(col)
    
    if return_classes:
        return pd.Series(le.transform(col)), list(le.classes_)
    else:
        return pd.Series(le.transform(col))


def one_hot_encode_column(col: pd.Series, return_classes=False):
    col_array = col.to_numpy()
    col_array = col_array.reshape((len(col), 1))
    print(col_array.shape)
    enc = OneHotEncoder()
    enc.fit(col_array)
    
    if return_classes:
        return pd.Series(enc.transform(col)), list(enc.categories_)
    else:
        return pd.Series(enc.transform(col))