"""Tests for utility functions in transformations.py"""

import unittest

import numpy as np
import pandas as pd

from dataframe_encoder import (
    trim_ions_string,
    trim_ions_ratio,
    CompositionEncoder,
    DFEncoder
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
        X = enc.transform(df)
        self.assertTupleEqual(X.shape, (5, 8))

    def test_dfencoder_fit_transform(self):
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
        X = enc.fit_transform(df)
        self.assertTupleEqual(X.shape, (5, 8))

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
        X = enc.fit_transform(df)
        # there is no duplicate row and one row with a nan value as target 
        # meaning the final number of rows will be one smaller than initially
        self.assertTupleEqual(X.shape, (5, 8))

if __name__ == '__main__':
    unittest.main()
