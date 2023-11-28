"""Using dataframe generated in clustering.py, plot unsupervised clusters."""

from absl import app
from absl import flags
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


FLAGS = flags.FLAGS

flags.DEFINE_string("file", "data/psc_tsne.csv", "Name of file to read.")
flags.DEFINE_string("target", "efficiency", "Name of target value to plot.")
flags.DEFINE_integer('font_size', 18, 'font size to use in labels')
flags.DEFINE_integer('tick_size', 16, 'font size to use in labels')


def main(argv):
    df = pd.read_csv(FLAGS.file, index_col=0)
    df = df.reindex(range(len(df)))
    print("Num rows: ", len(df))

    cols_print = [
    "electron_transport_layer", "hole_transport_layer",
    "back_contact", "substrate", "absorber_fabrication",
    "chemical_formula_descriptive", "efficiency", "efficiency_std", "band_gap"]

    def on_pick(event):
        artist = event.artist
        ind = event.ind
        print("Points selected: ", len(ind))
        for i in ind:
            print(df[cols_print].iloc[i])

    fig, ax = plt.subplots()
    tolerance = 1 # points
    cmap = mpl.cm.plasma
    #sns.scatterplot(df, x="tsne_0", y="tsne_1", hue=FLAGS.target, palette=cmap,
    #    ax=ax, picker=tolerance)
    cs = ax.scatter(df["tsne_0"], df["tsne_1"], c=df[FLAGS.target], picker=tolerance)
    fig.canvas.callbacks.connect('pick_event', on_pick)
    cbar = fig.colorbar(cs)
    ax.tick_params(which='both', labelsize=FLAGS.tick_size)
    ax.set_xlabel("TSNE X", fontsize=FLAGS.font_size)
    ax.set_ylabel("TSNE Y", fontsize=FLAGS.font_size)
    cbar.ax.set_ylabel(FLAGS.target, fontsize=FLAGS.font_size)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    app.run(main)