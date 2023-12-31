"""Do dimensional reduction and (un-)supervised learning using singular value
decomposition (SVD)."""

from absl import app
from absl import flags
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, GroupKFold
from sklearn.model_selection import cross_val_predict
#from sklearn.metrics import PredictionErrorDisplay

from transformations import get_compositions_vector_column

FLAGS = flags.FLAGS

flags.DEFINE_string("file", "example_data/psc_data.csv", "Name of file to read.")
flags.DEFINE_string("target", "efficiency", "Name of target value to fit.")
flags.DEFINE_bool("pairplot", False, "Set this flag to make a pairplot of svd features.")
flags.DEFINE_integer("n_dim", 5, "Number of dimensions of the truncated svd.")
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

    print(features_concat.shape)
    print(type(one_hot_features))
    #print("number of NaN values: ", np.sum(np.isnan(one_hot_features)))

    n_svd_components = FLAGS.n_dim
    svd = TruncatedSVD(n_components=n_svd_components, n_iter=10, random_state=42)
    svd.fit(features_concat)
    print(svd.explained_variance_ratio_)
    transformed_features = svd.transform(features_concat)
    print("number of NaN values: ", np.sum(np.isnan(transformed_features)))

    df_svd = pd.DataFrame(
        {i: list(transformed_features[:, i]) for i in range(n_svd_components)})
    df_svd[FLAGS.target] = df[FLAGS.target]
    df_svd = df_svd.dropna(subset=[FLAGS.target])
    #sns.scatterplot(df_svd, x=0, y=1, hue=FLAGS.target)
    #plt.show()
    if FLAGS.pairplot:
        sns.pairplot(df_svd, hue=FLAGS.target, kind='scatter')
        plt.show()

    # train a random forest on svd features
    regr = RandomForestRegressor(max_depth=100, random_state=0, max_features='sqrt', oob_score=True)
    #regr = SVR()
    cv = KFold(n_splits=5)
    invalid = np.isnan(df[FLAGS.target])
    targets = df_svd[FLAGS.target]
    inputs = df_svd[range(n_svd_components)].to_numpy()
    scores = cross_validate(regr, X=inputs, y=targets, cv=cv,
        scoring=['r2', 'neg_mean_absolute_error'], n_jobs=-1,
        return_estimator=True, error_score='raise')
    print(scores.keys())
    print(scores['test_r2'])
    print(-1*scores['test_neg_mean_absolute_error'])

    y_pred = cross_val_predict(regr, X=inputs, y=targets, cv=cv)
    
    df_regression = pd.DataFrame({FLAGS.target: targets, "prediction": y_pred})
    g = sns.JointGrid(
        data=df_regression, x=FLAGS.target, y='prediction', marginal_ticks=False
    )

    # Add the joint and marginal histogram plots
    g.plot_joint(
        sns.histplot, discrete=(False, False), bins=(50, 50),
    )
    g.plot_marginals(sns.histplot, element="step", color=None)
    g.ax_marg_x.set_xlabel('Count', fontsize=FLAGS.font_size)
    g.ax_marg_y.set_ylabel('Count', fontsize=FLAGS.font_size)
    g.ax_joint.tick_params(which='both', labelsize=FLAGS.tick_size)
    g.ax_joint.set_xlabel(FLAGS.target, fontsize=FLAGS.font_size)
    g.ax_joint.set_ylabel("prediction", fontsize=FLAGS.font_size)
    x_ref = np.linspace(*g.ax_joint.get_xlim())
    g.ax_joint.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    app.run(main)