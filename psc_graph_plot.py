"""Compute similarities between psc vectors and statistics of nearest neighbor
graph."""

from absl import app
from absl import flags
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import scipy
import networkx as nx

from transformations import get_compositions_vector_column

FLAGS = flags.FLAGS

flags.DEFINE_string("file", "example_data/psc_data.csv", "Name of file to read.")
flags.DEFINE_integer('font_size', 18, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 16, 'font size to use in labels')


def main(argv):
    df = pd.read_csv(FLAGS.file, index_col=0)

    df = df.dropna(subset=["chemical_formula_hill", "band_gap"])
    grouped = df.groupby(["chemical_formula_hill", "device_stack"], as_index=False)
    cols_categorical = [
        "electron_transport_layer", "hole_transport_layer",
        "back_contact", "substrate", "absorber_fabrication"]
    cols_numerical = [
        "band_gap", "efficiency", "open_circuit_voltage",
        "short_circuit_current_density", "fill_factor"]
    aggregators = ["mean", "std", "count"]
    agg_dict = {
        **{col: aggregators for col in cols_numerical},
        **{col: "first" for col in cols_categorical},
        "chemical_formula_hill": "first",
        "chemical_formula_descriptive": "first"
    }
    df_agg = grouped.agg(agg_dict)
    col_names = ['_'.join(col_name).strip().replace("_mean","").replace("_first","") \
        for col_name in df_agg.columns.values]
    df_agg.columns = col_names

    compositions = get_compositions_vector_column(
        df_agg["chemical_formula_hill"])
    comp_sparse = scipy.sparse.csr_matrix(compositions)

    enc = OneHotEncoder(sparse_output=True)
    one_hot_features = enc.fit_transform(df_agg[cols_categorical])

    minkowski_p = 1
    n_neighbors = 16

    fig, ax = plt.subplots(3, 1, sharex=True, layout='constrained')

    feature_list = [
        ('Composition+Stack', [comp_sparse, one_hot_features]),
        ('Composition only', [comp_sparse]),
        ('Stack only', [one_hot_features])
    ]
    for i, (title, feature_names) in enumerate(feature_list):
        x = scipy.sparse.hstack(feature_names)
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm="auto", p=minkowski_p).fit(x)
        dist, indices = nbrs.kneighbors(x)
        ax[i].hist(dist, bins=20, log=True)
        ax[i].set_title(title, fontsize=FLAGS.font_size)
        ax[i].tick_params(which='both', labelsize=FLAGS.tick_size)
        ax[i].set_ylabel("Count", fontsize=FLAGS.font_size)

    ax[2].set_xlabel(f'l-{minkowski_p} distance', fontsize=FLAGS.font_size)

    plt.tight_layout()
    plt.show()

    # make a networkx graph
    G = nx.DiGraph()

    print("Number of neighbors: ", n_neighbors)
    x = scipy.sparse.hstack([comp_sparse, one_hot_features])
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="auto", p=minkowski_p).fit(x)
    dist, indices = nbrs.kneighbors(x)

    for i in range(len(indices)):
        G.add_node(1)
        for j in indices[i]:
            G.add_edge(i, j)

    G = G.to_undirected()
    is_connected = nx.is_connected(G)
    print("Graph is connected: ", is_connected)
    if is_connected:
        print("Graph diameter: ", nx.diameter(G))
        print("Graph radius: ", nx.radius(G))


if __name__ == '__main__':
    app.run(main)
