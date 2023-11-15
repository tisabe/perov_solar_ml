"""Test the apriori algorithm on psc dataset."""

from absl import app
from absl import flags
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


FLAGS = flags.FLAGS
flags.DEFINE_string("file", "example_data/psc_data.csv", "Name of file to read.")


def main(argv):
    df = pd.read_csv(FLAGS.file, index_col=0)
    cols = [
        "substrate", "electron_transport_layer", "hole_transport_layer",
        "back_contact", "absorber_fabrication", "chemical_formula_descriptive"]
    df = df[cols]
    print("#rows with NA values: ", len(df) - len(df.dropna()))
    df = df.dropna()
    print(df.head())

    dataset = df.to_numpy()
    te = TransactionEncoder()
    data_enc = te.fit(dataset).transform(dataset)
    df_enc = pd.DataFrame(data_enc, columns=te.columns_)
    #print(df_enc.head())

    frequent_itemsets = apriori(df_enc, min_support=0.1, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(
        lambda x: len(x))
    print(len(frequent_itemsets))
    associations = association_rules(
        frequent_itemsets, metric="lift", min_threshold=0.9)
    print(associations[["antecedents", "consequents", "lift"]])


if __name__ == '__main__':
    app.run(main)