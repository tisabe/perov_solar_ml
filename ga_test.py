"""Test the genetic algorithm using PSC data and random forest regression to
optimize the power conversion efficiency of a hypothetical solar cell."""

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import pygad

from dataframe_encoder import (
    DFEncoder,
    aggregate_duplicate_rows,
    get_value_space
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "file", "example_data/psc_all_mz_20240109.csv", "Name of file to read.")


def main(argv):
    df = pd.read_csv(FLAGS.file, index_col=0)
    # define the columns we want to use
    # columns with lists of categories
    cols_cat_list = [
        #"Cell_stack_sequence",
        "ETL_deposition_procedure",
        "Substrate_stack_sequence",
        "ETL_stack_sequence",
        "ETL_deposition_procedure",
        "Perovskite_deposition_solvents",
        "HTL_stack_sequence",
        "HTL_deposition_procedure",
        "Backcontact_stack_sequence",
        "Backcontact_deposition_procedure"]
    # columns with single categories
    cols_category = [
        "Cell_architecture",
        "Perovskite_deposition_procedure",
        "Perovskite_deposition_aggregation_state_of_reactants"]
    # columns with lists of numeric values
    cols_num_list = [
        "Perovskite_dimension_list_of_layers",
        "Perovskite_deposition_thermal_annealing_temperature",
        "Perovskite_deposition_thermal_annealing_time"]
    # columns that determine the perovskite composition
    cols_composition = [
        "Perovskite_composition_a_ions_coefficients",
        "Perovskite_composition_b_ions_coefficients",
        "Perovskite_composition_c_ions_coefficients",
        "Perovskite_composition_a_ions",
        "Perovskite_composition_b_ions",
        "Perovskite_composition_c_ions",
    ]
    # columns with single numeric values
    cols_num = ["Cell_area_measured"]
    # columns with binary values
    cols_bin = [
        "Cell_flexible",
        "Cell_semitransparent",
        "Perovskite_single_crystal",
        "Perovskite_dimension_0D",
        "Perovskite_dimension_2D",
        "Perovskite_dimension_2D3D_mixture",
        "Perovskite_dimension_3D",
        "Perovskite_dimension_3D_with_2D_capping_layer",
        "Perovskite_deposition_quenching_induced_crystallisation",
        "Perovskite_deposition_solvent_annealing",
        "Encapsulation"]
    cols_all = [
        *cols_composition,
        *cols_cat_list,
        *cols_category,
        *cols_num_list,
        *cols_num,
        *cols_bin]

    cols_targets = ["JV_default_Voc", "JV_default_Jsc", "JV_default_PCE"]

    grouped = df.groupby(
        list(set(cols_all)),
        as_index=False)
    target = "JV_default_PCE"
    agg_dict = {
        target: ["mean", "std"],
        **{col: ["mean", "std"] for col in cols_num},
        **{col: "first" for col in cols_category},
        **{col: "first" for col in cols_cat_list},
        **{col: "first" for col in cols_num_list},
        **{col: "first" for col in cols_bin},
        **{col: "first" for col in cols_composition},
    }

    df_fit = grouped.agg(agg_dict)
    col_names = ['_'.join(col_name).strip().replace("_mean","").replace("_first","") \
        for col_name in df_fit.columns.values]
    df_fit.columns = col_names

    df_fit = df_fit.dropna(subset=target)
    print(df_fit.shape)
    composition_dict = {
        "Perovskite_composition_a_ions": "Perovskite_composition_a_ions_coefficients",
        "Perovskite_composition_b_ions": "Perovskite_composition_b_ions_coefficients",
        "Perovskite_composition_c_ions": "Perovskite_composition_c_ions_coefficients",
    }
    encoder = DFEncoder(
        target,
        cols_category+cols_cat_list+cols_num_list+cols_bin,
        composition_dict
    )
    df_out = encoder.fit_transform(df_fit, append=False)
    print(df_out.shape)


if __name__ == '__main__':
    app.run(main)
