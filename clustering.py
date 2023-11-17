"""Try different clustering algorithms."""

from absl import app
from absl import flags
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, GroupKFold
from sklearn.model_selection import cross_val_predict
from sklearn.manifold import Isomap, TSNE

from transformations import get_compositions_vector_column


FLAGS = flags.FLAGS

flags.DEFINE_string("file", "example_data/psc_data.csv", "Name of file to read.")
flags.DEFINE_string("target", "efficiency", "Name of target value to plot.")
flags.DEFINE_integer("n_dim", 2, "Number of dimensions of the reduced data.")
flags.DEFINE_integer('font_size', 18, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 16, 'font size to use in labels')


def main(argv):
    df = pd.read_csv(FLAGS.file, index_col=0)

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
    #print(grouped.keys)
    #grouped = grouped.count()
    df_agg = grouped.agg(agg_dict)
    # flatten and strip column index levels
    col_names = ['_'.join(col_name).strip().replace("_mean","").replace("_first","") \
        for col_name in df_agg.columns.values]
    df_agg.columns = col_names
    df_agg = df_agg.dropna(subset=["band_gap"])

    enc = OneHotEncoder(sparse=True#, min_frequency=100,
        ) # change keyword to sparse_output for sklearn version 1.2 or later
    enc.fit(df_agg[cols_categorical])
    one_hot_features = np.array(enc.transform(df_agg[cols_categorical]).todense())

    compositions = get_compositions_vector_column(
        df_agg["chemical_formula_hill"])

    band_gaps = df_agg['band_gap'].to_numpy().reshape((-1,1))
    features_concat = np.concatenate(
        (compositions, one_hot_features, band_gaps), axis=1)

    n_dim = FLAGS.n_dim
    #enc = TruncatedSVD(n_components=n_dim, n_iter=10, random_state=42)
    #enc = Isomap(n_components=n_dim)
    enc = PCA(n_components=100)
    enc.fit(features_concat)
    try:
        print("Encoder explained variances: ")
        print(enc.explained_variance_ratio_)
    except AttributeError:
        print("No explained_variance_ratio_ attribute in this encoder.")
    reduced_features = enc.transform(features_concat)
    print("number of NaN values: ", np.sum(np.isnan(reduced_features)))
    #sns.scatterplot(df, x="svd_0", y="svd_1", hue=col_example)
    #plt.show()

    tsne = TSNE(n_components=2, learning_rate='auto')
    transformed_features = tsne.fit_transform(reduced_features)

    for i, feature_column in enumerate(transformed_features.T):
        df_agg["tsne_"+str(i)] = feature_column
    df_agg.to_csv("data/psc_tsne.csv")
    sns.scatterplot(df_agg, x="tsne_0", y="tsne_1", hue="efficiency")
    plt.show()



if __name__ == '__main__':
    app.run(main)