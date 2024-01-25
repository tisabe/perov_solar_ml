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
    get_value_space,
    filter_compositions,
    filter_singlelayer,
    CompositionEncoder_DF,
    filter_common
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
        df = pd.DataFrame({
            "target": [1., 2., 1., 2., 3., 4.],
            "cat0": ["A", "B", "C", "A", "B", "B"],
            "cat1": ["C", "C", "D", "D", "D", "D"],
            "comp_a": ["a | b | c", "a; b; c", "a", "a", "a", None],
            "comp_a_coefficient": ["8 | 1 | 1", "0.5; 0.1; 0.4", "1", "1", None, "1"]},
            index=([0, 1, 4, 5, 10, 11])
        )
        enc = CompositionEncoder()
        enc.fit(df["comp_a"])
        df_compositions = enc.transform(
            df["comp_a"], df["comp_a_coefficient"])
        df_compositions_expected = pd.DataFrame({
            "a": [.8, .5, 1., 1., None, None],
            "b": [.1, .1, 0., 0., None, None],
            "c": [.1, .4, 0., 0., None, None]},
            index=([0, 1, 4, 5, 10, 11]))
        pd.testing.assert_frame_equal(
            df_compositions, df_compositions_expected)

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

    def test_filter_compositions(self):
        df = pd.DataFrame({
            "comp_a": ["MA", "Cs; FA; MA", "MA", "MA"],
            "comp_b": ["Pb", "Pb", "Pb | Sn", "Sn; Ti"]})
        filter_dict = {"comp_a": ["MA"], "comp_b": ["Sn", "Pb"]}
        df_filtered = filter_compositions(df, filter_dict)
        df_expected = pd.DataFrame({
            "comp_a": ["MA", "MA"],
            "comp_b": ["Pb", "Pb | Sn"]}, index=[0, 2])
        pd.testing.assert_frame_equal(df_filtered, df_expected)

    def test_filter_singlelayer(self):
        df = pd.DataFrame({
            "comp_a": ["MA", "Cs; FA; MA", "MA", "MA"],
            "comp_b": ["Pb", "Pb", "Pb | Sn", "Sn; Ti"]})
        cols_comp = ["comp_a", "comp_b"]
        df_filtered = filter_singlelayer(df, cols_comp)
        df_expected = pd.DataFrame({
            "comp_a": ["MA", "Cs; FA; MA", "MA"],
            "comp_b": ["Pb", "Pb", "Sn; Ti"]}, index=[0, 1, 3])
        pd.testing.assert_frame_equal(df_filtered, df_expected)

    def test_CompositionEncoder_DF(self):
        df = pd.DataFrame({
            "target": [1., None, 1., 2., 3., 4.],
            "cat0": [None, "B", "C", "A", "B", "B"],
            "cat1": ["C", "C", "D", "D", "D", "D"],
            "comp_a": ["a | b | c", "a; b; c", "a", "a", "a", "c"],
            "comp_a_coefficient": ["8 | 1 | 1", "0.5; 0.1; 0.4", "1", "1", "1", "1"],
            "comp_b": ["A | B; C", "A", "A", "A", "A", "A"],
            "comp_b_coefficient": ["1 | 0.5; 0.5", "0.5", "1", "1", "1", "1"]})
        composition_dict = {
            "comp_a": "comp_a_coefficient",
            "comp_b": "comp_b_coefficient"}
        enc = CompositionEncoder_DF(composition_dict)
        enc.fit(df)
        df_comp = enc.transform(df)
        df_expected = pd.DataFrame({
            "a": [.8, .5, 1., 1., 1., 0.],
            "b": [.1, .1, 0., 0., 0., 0.],
            "c": [.1, .4, 0., 0., 0., 1.],
            "A": [.5, 1., 1., 1., 1., 1.],
            "B": [.25, .0, .0, .0, .0, 0.],
            "C": [.25, .0, .0, .0, .0, 0.],
        })
        pd.testing.assert_frame_equal(df_comp, df_expected)

        df_comp = enc.fit_transform(df)
        pd.testing.assert_frame_equal(df_comp, df_expected)

    def test_CompositionEncoder_DF_raises(self):
        df = pd.DataFrame({
            "comp_a": ["a; b", "c"],
            "comp_a_coefficient": ["0.5; 0.5", "1."],
            "comp_b": ["A", "c; C"],
            "comp_b_coefficient": ["1.", "0.5; 0.5"]})
        composition_dict = {
            "comp_a": "comp_a_coefficient",
            "comp_b": "comp_b_coefficient"}
        enc = CompositionEncoder_DF(composition_dict)
        with self.assertRaises(ValueError):
            enc.fit(df)

    def test_filter_common(self):
        df = pd.DataFrame({
            "a": [1]*12 + [2]*8,
            "b": [3]*5 + [4]*15,
            "c": [1]*20,
            "d": range(20)},
            index=range(4, 24)
        )
        df_common = filter_common(df, cols=["a", "b", "c"], thresh=0.5)
        df_expected = df.loc[9:15]
        pd.testing.assert_frame_equal(df_common, df_expected)


if __name__ == '__main__':
    unittest.main()
