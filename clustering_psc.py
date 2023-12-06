"""Requires scikit-learn >=1.3 for HDBSCAN."""
import time
import warnings
from collections import defaultdict
from itertools import cycle, islice

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn import metrics
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import MDS

from transformations import get_compositions_vector_column


def segment_sum(data, segment_ids):
    data = np.asarray(data)
    s = np.zeros((np.max(segment_ids)+1,) + data.shape[1:], dtype=data.dtype)
    np.add.at(s, segment_ids, data)
    return s

def var_score(targets, cluster_labels):
    seg_sum = segment_sum(targets, cluster_labels)
    _, counts = np.unique(cluster_labels, return_counts=True)
    seg_mean = seg_sum/counts
    abs_diffs = np.abs(targets - seg_mean[cluster_labels])
    var = np.square(abs_diffs)
    return np.mean(var)

np.random.seed(0)

### Load and prepare the data
df = pd.read_csv("example_data/psc_data.csv", index_col=0)

cols_categorical = [
    "electron_transport_layer", "hole_transport_layer",
    "back_contact", "substrate", "absorber_fabrication"]
cols_numerical = [
    "band_gap", "efficiency", "open_circuit_voltage",
    "short_circuit_current_density", "fill_factor"]
cols_optional_num = [
    "device_area", "illumination_intensity"]
aggregators = ["mean", "std", "count"]
agg_dict = {
    **{col: aggregators for col in cols_numerical},
    **{col: "first" for col in cols_categorical},
    "chemical_formula_hill": "first",
    "chemical_formula_descriptive": "first"
}
# drop all columns where there is no composition defined
df = df.dropna(subset=["chemical_formula_hill"])
#df = df.astype({col: str for col in cols_categorical})

# grouping cells with same device stack and formula together
grouped = df.groupby(cols_categorical+["chemical_formula_hill"], as_index=False)
df_agg = grouped.agg(agg_dict)
# flatten and strip column index levels
col_names = ['_'.join(col_name).strip().replace("_mean","").replace("_first","") \
    for col_name in df_agg.columns.values]
df_agg.columns = col_names
df_agg = df_agg.dropna(subset=["band_gap", "efficiency"])

targets = np.reshape(df_agg["efficiency"].to_numpy(), (-1,1))
print(targets)
discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
targets = discretizer.fit_transform(targets)
print("Efficiency bins: ", discretizer.bin_edges_)

enc = OneHotEncoder(sparse_output=True#, min_frequency=100,
    )
enc.fit(df_agg[cols_categorical])
one_hot_features = np.array(enc.transform(df_agg[cols_categorical]).todense())

compositions = get_compositions_vector_column(
    df_agg["chemical_formula_hill"])

band_gaps = df_agg['band_gap'].to_numpy().reshape((-1,1))
features_concat = np.concatenate(
    (compositions, one_hot_features, band_gaps), axis=1)
print("Shape of featurized data: ", features_concat.shape)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 13))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)

plot_num = 1

seed = 42

n_components = 2
pca = PCA(n_components=n_components)
svd = TruncatedSVD(n_components=n_components)
mds = MDS(n_components=n_components)
dim_reductions = (
    #("identity", None),
    ("pca", pca),
    ("svd", svd),
    ("mds", mds)
)

n_clusters = 10
allow_single_cluster = True
min_samples = 10
min_cluster_size = 10
hdbscan_min_samples = 10
hdbscan_min_cluster_size = 15
xi = 0.05

metrics = {
    "rand_score": metrics.adjusted_rand_score,
    "mutual_info": metrics.adjusted_mutual_info_score,
    "homogeneity": metrics.homogeneity_score,
    "completeness": metrics.completeness_score,
    "silhouette": metrics.silhouette_score,
}
scores = defaultdict(list)

X, y = features_concat, np.reshape(targets, (-1,))

for i_reduction, (name_reduction, fun_reduction) in enumerate(dim_reductions):
    print(name_reduction)

    if name_reduction == "identity":
        X_tr = X
    else:
        X_tr = fun_reduction.fit_transform(X)
        for dim_i in range(n_components):
            df_agg[name_reduction+"_"+str(dim_i)] = X_tr[:, dim_i]
    print("Shape of transformed data: ", X_tr.shape)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X_tr, n_neighbors=5, include_self=False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init="auto",
        random_state=seed,
    )
    ward = cluster.AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
        connectivity=connectivity
    )
    hdbscan = cluster.HDBSCAN(
        min_samples=hdbscan_min_samples,
        min_cluster_size=hdbscan_min_cluster_size,
        allow_single_cluster=allow_single_cluster
    )
    optics = cluster.OPTICS(
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
    )
    gmm = mixture.GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=seed,
    )

    clustering_algorithms = (
        ("kmeans", kmeans),
        ("ward", ward),
        ("hdbscan", hdbscan),
        ("optics", optics),
        ("gmm", gmm),
    )

    for name, algorithm in clustering_algorithms:
        print(name)
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                + "connectivity matrix is [0-9]{1,2}"
                + " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding"
                + " may not work as expected.",
                category=UserWarning,
            )
            algorithm.fit(X_tr)

        t1 = time.time()
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X_tr)

        # calculate clustering performance evaluation
        for score_name, fun_score in metrics.items():
            if score_name == "silhouette":
                score = fun_score(X_tr, y_pred, metric='euclidean')
            else:
                score = fun_score(y, y_pred)
            scores["score"].append(score)
            scores["score_name"].append(score_name)
            scores["reduction"].append(name_reduction)
            scores["method"].append(name)

        plt.subplot(len(dim_reductions), len(clustering_algorithms), plot_num)
        if i_reduction == 0:
            plt.title(name, size=18)

        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xticks(())
        plt.yticks(())
        plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )
        plot_num += 1
df = pd.DataFrame(scores)
print(df)
df_agg.to_csv("data/psc_cluster.csv")

plt.show()

sns.barplot(df, x="method", y="score", hue="score_name")
plt.show()
