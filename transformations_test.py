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

    def test_trim_stack_string(self):
        stacks = self.df["device_stack"]
        for stack in stacks[:10]:
            stack_list = trim_stack_string(stack)
            #print(stack_list)
            #print(len(stack_list))

        def get_stack_len(stack_string):
            return len(trim_stack_string(stack_string))

        layers_all = np.array([], dtype=str)
        max_len = 0
        for stack in self.df["device_stack"]:
            stack_len = get_stack_len(stack)
            if stack_len > max_len:
                max_len = stack_len
                print(stack)

        stack_lens = self.df["device_stack"].apply(get_stack_len)
        print(max(stack_lens))
        print(min(stack_lens))

        stacks = self.df["device_stack"].apply(trim_stack_string)
        layers_all = []
        for stack in stacks:
            for layer in stack:
                layers_all.append(layer)
        print(len(layers_all))
        print(len(set(layers_all)))
        print(set(layers_all))

    def test_StackEncoder(self):
        enc = StackEncoder()
        stacks = self.df["device_stack"].to_list()
        enc.fit(stacks)

        layers_all = []
        for stack in stacks:
            stack = trim_stack_string(stack)
            for layer in stack:
                layers_all.append(layer)
        print(enc.enc.categories_[0])
        self.assertEqual(len(set(layers_all)), len(enc.enc.categories_[0]))

        stacks_tr = enc.transform(stacks[:])
        print(stacks_tr)
        print(stacks_tr.shape)
        print(np.sum(stacks_tr))


if __name__ == '__main__':
    unittest.main()
