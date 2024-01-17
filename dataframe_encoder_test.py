"""Tests for utility functions in transformations.py"""

import unittest

import numpy as np
import pandas as pd

from dataframe_encoder import (
    trim_ions_string,
    trim_ions_ratio,
    CompositionEncoder,
    DFEncoder,
    aggregate_duplicate_rows,
    get_value_space
)


class UnitTest(unittest.TestCase):
    """Testing class for utility functions."""
    def setUp(self):
        pass

    def test_trim_ions_string(self):
        test_str = "a | b; c | d; e | f | g"
        trimmed = trim_ions_string(test_str)
        expected = ["a", "b", "c", "d", "e", "f", "g"]
        self.assertListEqual(trimmed, expected)

    def test_trim_ions_ratio(self):
        test_str = "0.4 | 0.5; 0.6 | 0.7; 0.8 | 1.0 | 1"
        trimmed = trim_ions_ratio(test_str)
        expected = [0.4, 0.5, 0.6, 0.7, 0.8, 1., 1.]
        self.assertListEqual(trimmed, expected)
    
    def test_composition_encoder(self):
        ions_stack_strings = [
            "a | b | c",
            "a; b; c",
            "a | b; c",
            "a",
            None,
            "b"
        ]
        ions_stack_ratios = [
            "8 | 1 | 1",
            "0.5; 0.1; 0.4",
            "1 | 0.5; 0.5",
            "0.5",
            "1",
            None
        ]
        enc = CompositionEncoder()
        enc.fit(ions_stack_strings)
        compositions = enc.transform(ions_stack_strings, ions_stack_ratios)
        compositions_expected = np.array([
            [0.8, 0.1, 0.1],
            [0.5, 0.1, 0.4],
            [0.5, 0.25, 0.25],
            [1., 0., 0.],
            [np.nan]*3,
            [np.nan]*3
        ])
        np.testing.assert_array_equal(compositions, compositions_expected)
    
    def test_dfencoder_fit_and_transform(self):
        cols_target = "target"
        cols_category = ["cat0", "cat1"]
        cols_composition_dict = {
            "comp_a": "comp_a_coefficient", "comp_b": "comp_b_coefficient"}
        enc = DFEncoder(
            cols_target,
            cols_category,
            cols_composition_dict
        )
        df = pd.DataFrame({
            "target": [1., 2., 1., 2., 3., 4.],
            "cat0": ["A", "B", "C", "A", "B", "B"],
            "cat1": ["C", "C", "D", "D", "D", "D"],
            "comp_a": ["a | b | c", "a; b; c", "a", "a", "a", "a"],
            "comp_a_coefficient": ["8 | 1 | 1", "0.5; 0.1; 0.4", "1", "1", "1", "1"],
            "comp_b": ["a | b; c", "a", "a", "a", "a", "a"],
            "comp_b_coefficient": ["1 | 0.5; 0.5", "0.5", "1", "1", "1", "1"]
        })
        enc.fit(df)
        df_append = enc.transform(df, append=True)
        self.assertTupleEqual(df_append.shape, (6, 15))
        self.assertListEqual(
            ['target', 'cat0', 'cat1', 'comp_a', 'comp_a_coefficient',
            'comp_b', 'comp_b_coefficient', 'comp_a_a', 'comp_a_b', 'comp_a_c',
            'comp_b_a', 'comp_b_b', 'comp_b_c', 'cat0_out', 'cat1_out'],
            list(df_append.columns.values))
        
        df_out = enc.transform(df, append=False)
        self.assertTupleEqual(df_out.shape, (6, 8))
        self.assertListEqual(
            ['comp_a_a', 'comp_a_b', 'comp_a_c', 'comp_b_a', 'comp_b_b',
            'comp_b_c', 'cat0_out', 'cat1_out'],
            list(df_out.columns.values))

    def test_dfencoder_fit_transform(self):
        cols_target = "target"
        cols_category = ["cat0", "cat1"]
        cols_composition_dict = {
            "comp_a": "comp_a_coefficient", "comp_b": "comp_b_coefficient"}
        df = pd.DataFrame({
            "target": [1., 2., 1., 2., 3., 4.],
            "cat0": ["A", "B", "C", "A", "B", "B"],
            "cat1": ["C", "C", "D", "D", "D", "D"],
            "comp_a": ["a | b | c", "a; b; c", "a", "a", "a", "a"],
            "comp_a_coefficient": ["8 | 1 | 1", "0.5; 0.1; 0.4", "1", "1", "1", "1"],
            "comp_b": ["a | b; c", "a", "a", "a", "a", "a"],
            "comp_b_coefficient": ["1 | 0.5; 0.5", "0.5", "1", "1", "1", "1"]
        })
        enc = DFEncoder(
            cols_target,
            cols_category,
            cols_composition_dict
        )
        df_out = enc.fit_transform(df, append=True)
        self.assertTupleEqual(df_out.shape, (6, 15))
        self.assertListEqual(
            ['target', 'cat0', 'cat1', 'comp_a', 'comp_a_coefficient',
            'comp_b', 'comp_b_coefficient', 'comp_a_a', 'comp_a_b', 'comp_a_c',
            'comp_b_a', 'comp_b_b', 'comp_b_c', 'cat0_out', 'cat1_out'],
            list(df_out.columns.values))
        enc = DFEncoder(
            cols_target,
            cols_category,
            cols_composition_dict
        )
        df_out = enc.fit_transform(df, append=False)
        self.assertTupleEqual(df_out.shape, (6, 8))
        self.assertListEqual(
            ['comp_a_a', 'comp_a_b', 'comp_a_c', 'comp_b_a', 'comp_b_b',
            'comp_b_c', 'cat0_out', 'cat1_out'],
            list(df_out.columns.values))

    def test_dfencoder_nan_vals(self):
        cols_target = "target"
        cols_category = ["cat0", "cat1"]
        cols_composition_dict = {
            "comp_a": "comp_a_coefficient", "comp_b": "comp_b_coefficient"}
        enc = DFEncoder(
            cols_target,
            cols_category,
            cols_composition_dict
        )
        df = pd.DataFrame({
            "target": [1., None, 1., 2., 3., 4.],
            "cat0": [None, "B", "C", "A", "B", "B"],
            "cat1": ["C", "C", "D", "D", "D", "D"],
            "comp_a": ["a | b | c", "a; b; c", "a", "a", "a", "c"],
            "comp_a_coefficient": ["8 | 1 | 1", "0.5; 0.1; 0.4", "1", "1", "1", "1"],
            "comp_b": ["a | b; c", "a", "a", "a", "a", "a"],
            "comp_b_coefficient": ["1 | 0.5; 0.5", "0.5", "1", "1", "1", "1"]
        })
        # NaN target values rase a value error and have to be filtered out
        with self.assertRaises(ValueError):
            df_out = enc.fit_transform(df, append=False)
        df = df.dropna(subset="target")
        df_out = enc.fit_transform(df, append=False)
        self.assertTupleEqual(df_out.shape, (5, 8)) # one was deleted

    def test_aggregate_duplicate_rows(self):
        df = pd.DataFrame({
            'id':['a', 'b', 'c', 'd', 'e'],
            'A': [1, 1, 2, 2, None],
            'B': [1, 2, 2, 2, 1],
            'c': [1., 2., 3., 3., 4.]
        })
        df_agg = aggregate_duplicate_rows(df, ['A', 'B'], ['c'], dropna=False)
        # TODO: for some reason the column order changes, find out why
        df_expected = pd.DataFrame({
            'c': [1., 2., 3., 4.],
            'c_std': [None, None, 0., None],
            'A': [1, 1, 2, None],
            'B': [1, 2, 2, 1],
            'id': ['a', 'b', 'c', 'e']})
        pd.testing.assert_frame_equal(df_agg, df_expected)
        # test function with dropna=True
        df = pd.DataFrame({
            'id':['a', 'b', 'c', 'd', 'e'],
            'A': [1, 1, 2, 2, None],
            'B': [1, 2, 2, 2, 1],
            'c': [1., 2., 3., 3., 4.]
        })
        df_agg = aggregate_duplicate_rows(df, ['A', 'B'], ['c'], dropna=True)
        df_expected = pd.DataFrame({
            'c': [1., 2., 3.],
            'c_std': [None, None, 0.],
            'A': [1., 1., 2.],
            'B': [1, 2, 2],
            'id': ['a', 'b', 'c']})
        pd.testing.assert_frame_equal(df_agg, df_expected)

    def test_get_value_space(self):
        df = pd.DataFrame({
            'id':['a', 'b', 'c', 'd', 'e'],
            'A': [1, 1, 2, 2, None],
            'B': ['1', '2', '2', '2', '1'],
            'c': [1., 2., 3., 3., 4.]
        })
        cols_type_dict = {'A': 'int', 'B': 'str', 'c': np.float64}
        value_space = get_value_space(df, cols_type_dict)
        value_space_expected = [
            {'low': 1.0, 'high': 2.0, 'step': 1},
            ['1', '2'],
            {'low': 1.0, 'high': 4.0}]
        self.assertListEqual(value_space, value_space_expected)
        # with dropna=True, drops the fourth row
        value_space = get_value_space(df, cols_type_dict, dropna=True)
        value_space_expected = [
            {'low': 1.0, 'high': 2.0, 'step': 1},
            ['1', '2'],
            {'low': 1.0, 'high': 3.0}]
        self.assertListEqual(value_space, value_space_expected)



if __name__ == '__main__':
    unittest.main()
