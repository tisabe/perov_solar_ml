from ase.formula import Formula
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

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