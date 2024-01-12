# Methods for encoding a dataframe as produced from the Perovskite Database
import numpy as np
from sklearn.preprocessing import TargetEncoder


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

    def fit(self, ions_stack_strings):
        ions_stacks = [trim_ions_string(ions_stack_string) for ions_stack_string in ions_stack_strings]
        self.ions_unique = []
        for ions_stack in ions_stacks:
            for ion in ions_stack:
                self.ions_unique.append(ion)
        self.ions_unique = list(set(self.ions_unique))
        self.n_categories = len(self.ions_unique)

    def transform(self, ions_names, ions_ratios):
        ions_stacks = [trim_ions_string(comp) for comp in ions_names]
        ions_ratios = [trim_ions_ratio(comp) for comp in ions_ratios]

        compositions = []
        for ions, ratios in zip(ions_stacks, ions_ratios):
            composition = np.zeros(self.n_categories)
            indices = [self.ions_unique.index(ion) for ion in ions]
            for index, ratio in zip(indices, ratios):
                composition[index] += ratio
            composition /= np.sum(composition)
            compositions.append(composition)

        return np.vstack(compositions)


class DFEncoder:
    def __init__(self):
        self.features = []
    
    def fit(self, df):
        pass

    def transform(self, df):
        pass
