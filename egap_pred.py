"""Predict electronic band gaps from elemental composition."""

from absl import app
from absl import flags
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, GroupKFold
from sklearn.model_selection import cross_val_predict
#from sklearn.metrics import PredictionErrorDisplay

from transformations import get_compositions_vector_column


FLAGS = flags.FLAGS

flags.DEFINE_string("file", "example_data/psc_data.csv", "Name of file to read.")
flags.DEFINE_integer('font_size', 18, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 16, 'font size to use in labels')


def main(argv):
    df = pd.read_csv(FLAGS.file, index_col=0)

    df = df.dropna(subset=["chemical_formula_hill", "band_gap"])
    grouped = df.groupby(["chemical_formula_hill"], as_index=False)
    cols_numerical = [
        "band_gap", "efficiency", "open_circuit_voltage",
        "short_circuit_current_density", "fill_factor"]
    aggregators = ["mean", "std", "count"]
    agg_dict = {
        **{col: aggregators for col in cols_numerical},
        "chemical_formula_hill": "first",
        "chemical_formula_descriptive": "first"
    }
    df_agg = grouped.agg(agg_dict)
    col_names = ['_'.join(col_name).strip().replace("_mean","").replace("_first","") \
        for col_name in df_agg.columns.values]
    df_agg.columns = col_names

    compositions = get_compositions_vector_column(
        df_agg["chemical_formula_hill"])
    print("Shape of compositions data: ", compositions.shape)
    # train a random forest on svd features
    regr = RandomForestRegressor(max_depth=100, random_state=0, max_features='sqrt', oob_score=True)
    #regr = SVR()
    #regr = LinearRegression()
    cv = KFold(n_splits=5)
    targets = df_agg["band_gap"]
    inputs = compositions
    metrics = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
    metrics_abbrev = {"r2": "r2", "neg_mean_absolute_error": "-MAE",
        "neg_root_mean_squared_error": "-RMSE"}
    scores = cross_validate(regr, X=inputs, y=targets, cv=cv,
        scoring=metrics, n_jobs=-1,
        return_estimator=True, error_score='raise')

    for metric in metrics:
        score = scores["test_"+metric]
        print(metrics_abbrev[metric], " score: ", score)
        print("Averaged: ", np.mean(score), "+-", np.std(score))

    print("Average number of cells per composition: ",
        np.mean(df_agg["band_gap_count"]))
    print("Average standard deviation of band gap per composition: ",
        np.mean(df_agg["band_gap_std"]))

    y_pred = cross_val_predict(regr, X=inputs, y=targets, cv=cv)
    
    df_agg["prediction"] = y_pred
    g = sns.JointGrid(
        data=df_agg, x="band_gap", y='prediction', marginal_ticks=False
    )

    # Add the joint and marginal histogram plots
    g.plot_joint(
        sns.histplot, discrete=(False, False), bins=(50, 50),
    )
    g.plot_marginals(sns.histplot, element="step", color=None)
    g.ax_marg_x.set_xlabel('Count', fontsize=FLAGS.font_size)
    g.ax_marg_y.set_ylabel('Count', fontsize=FLAGS.font_size)
    g.ax_joint.tick_params(which='both', labelsize=FLAGS.tick_size)
    g.ax_joint.set_xlabel("band_gap", fontsize=FLAGS.font_size)
    g.ax_joint.set_ylabel("prediction", fontsize=FLAGS.font_size)
    x_ref = np.linspace(*g.ax_joint.get_xlim())
    g.ax_joint.plot(x_ref, x_ref, '--', alpha=0.2, color='grey')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    app.run(main)