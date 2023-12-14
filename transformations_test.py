"""Tests for utility functions in transformations.py"""

import unittest

import numpy as np
import pandas as pd

from transformations import (
    one_hot_encode_column,
    trim_stack_string,
    StackEncoder
)


class UnitTest(unittest.TestCase):
    """Testing class for utility functions."""
    def setUp(self):
        self.df = pd.read_csv("example_data/psc_data.csv", index_col=0)

    def test_trim_stack_string_db(self):
        num_stacks = None
        stacks = self.df["device_stack"].to_list()[:num_stacks]
        for stack in stacks:
            stack_list = trim_stack_string(stack)

    def test_trim_stack_string(self):
        stacks = [
            "['C', 'A', 'A', 'A', 'A']",
            "['A', 'B', 'B', 'A']",
            "['A', 'D; B', '']"
        ]
        stacks_expected = [
            ['C', 'A', 'A', 'A', 'A'],
            ['A', 'B', 'B', 'A'],
            ['A', 'D; B', '']
        ]
        for stack, stack_expected in zip(stacks, stacks_expected):
            stack = trim_stack_string(stack)
            self.assertEqual(stack, stack_expected)

    def test_StackEncoder_db(self):
        """Test the StackEncoder class by fitting, transforming and inverse
        transforming the device stacks of the database."""
        num_stacks = None
        enc = StackEncoder(min_frequency=None)
        stacks = self.df["device_stack"].to_list()[:num_stacks]
        enc.fit(stacks)

        layers_all = []
        for stack in stacks:
            stack = trim_stack_string(stack)
            for layer in stack:
                layers_all.append(layer)
        self.assertEqual(len(set(layers_all)), len(enc.enc.categories_[0]))

        stacks_tr = enc.transform(stacks)
        self.assertEqual(len(layers_all), np.sum(stacks_tr))
        unique, counts = np.unique(stacks_tr, return_counts=True)

        stacks_inverse = enc.inverse_transform(stacks_tr)
        for stack, stack_inverse in zip(stacks, stacks_inverse):
            stack = trim_stack_string(stack)
            self.assertEqual(set(stack), set(stack_inverse))

    def test_StackEncoder_min_frequency(self):
        """Test the min_frequency parameter of the StackEncoder."""
        enc = StackEncoder(min_frequency=2)
        stacks = [
            "['C', 'A', 'A', 'A', 'A']",
            "['A', 'B', 'B', 'A']",
            "['A', 'D', 'A']"
        ]
        stacks_tr_expected = np.array([
            [4, 0, 1],
            [2, 2, 0],
            [2, 0, 1]
        ])
        enc.fit(stacks)
        stacks_tr = enc.transform(stacks)
        np.testing.assert_array_equal(stacks_tr, stacks_tr_expected)

        stacks_inverse = enc.inverse_transform(stacks_tr)
        category_transform = {
            'A': 'A', 'B': 'B', 'C': 'infrequent_sklearn',
            'D': 'infrequent_sklearn'}
        for stack, stack_inverse in zip(stacks, stacks_inverse):
            stack = trim_stack_string(stack)
            stack = [category_transform[layer] for layer in stack]
            self.assertEqual(set(stack), set(stack_inverse))

if __name__ == '__main__':
    unittest.main()
