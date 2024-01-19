"""Test the genetic algorithm using PSC data and random forest regression to
optimize the power conversion efficiency of a hypothetical solar cell."""

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import pygad
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_validate
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder
from sklearn.svm import SVR
import matplotlib.pyplot as plt

from dataframe_encoder import (
    DFEncoder,
    aggregate_duplicate_rows,
    get_value_space
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "file", "example_data/psc_all_mz_20240109.csv", "Name of file to read.")


def main(argv):
    df = pd.read_csv(FLAGS.file, index_col=0, low_memory=False)
    # define the columns we want to use
    # columns with lists of categories
    cols_cat_list = [
        #"Cell_stack_sequence",
        "ETL_deposition_procedure",
        "Substrate_stack_sequence",
        "ETL_stack_sequence",
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
    print("Data shape after aggregation: ", df_fit.shape)
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
    print("Data shape after encoding: ", df_out.shape)
    #print(df_out.columns.values)

    cols_composition_new = [col for col in df_out.columns.values \
        if "Perovskite_composition_" in col]
    #print(cols_composition_new)
    print("Composition cols df shape: ", df_out[cols_composition_new].shape)
    print("Categorical+Numerical df shape (before encoding): ",
        df_fit[cols_category+cols_cat_list+cols_num_list+cols_bin].shape)

    regr = RandomForestRegressor(max_depth=100, random_state=0,
        max_features='sqrt', oob_score=True, n_jobs=-1)
    #regr = SVR()
    X_combined = df_out.to_numpy()
    y = df_fit[target].to_numpy()
    cv = KFold(n_splits=5)
    scores = cross_validate(regr, X=X_combined, y=y, cv=cv,
        scoring=['r2', 'neg_mean_absolute_error'], n_jobs=-1,
        return_estimator=True, error_score='raise')
    print("Scoring attributes: ", scores.keys())
    print("r^2: ", scores['test_r2'])
    print("MAE: ", -1*scores['test_neg_mean_absolute_error'])
    print("Average standard deviation of the target: ",
        df_fit[target+"_std"].mean())

    y_pred = cross_val_predict(regr, X=X_combined, y=y, cv=cv)

    # collect the types of different columns to get the gene space
    cols_type_dict_categories = {
        **{col: "int" for col in cols_category},
        **{col: "int" for col in cols_cat_list},
        **{col: "int" for col in cols_num_list},
        **{col: "int" for col in cols_bin},
    }
    enc_ordinal = OrdinalEncoder()
    X_ordinal = enc_ordinal.fit_transform(
        df_fit[cols_type_dict_categories.keys()])
    #print(enc_ordinal.feature_names_in_, len(enc_ordinal.feature_names_in_))
    df_ordinal_cat = pd.DataFrame(
        X_ordinal, columns=list(cols_type_dict_categories.keys()))
    value_space_cat = get_value_space(df_ordinal_cat, cols_type_dict_categories)
    cols_type_dict_nums = {
        #**{col: "float" for col in cols_num},
        **{col: "float" for col in cols_composition_new}}
    value_space_num = get_value_space(df_out, cols_type_dict_nums)
    value_space = value_space_cat+value_space_num

    enc_cat = encoder.enc_target
    num_cat_cols = len(cols_type_dict_categories.keys())

    def fitness_func_batch(ga, solution, solution_idx):
        # decompose the solution into composition part and categorical part
        solution_str = enc_ordinal.inverse_transform(
            solution[:, :num_cat_cols])
        solution_str_df = pd.DataFrame(
            solution_str, columns=cols_type_dict_categories.keys())
        categorical_vec = enc_cat.transform(solution_str_df)
        composition_vec = solution[:, num_cat_cols:]

        rf_input = np.concatenate(
            (categorical_vec, composition_vec), axis=1)
        # we get k predictions as we used k-fold cross validation earlier
        y_preds = [estimator.predict(rf_input) for estimator in scores['estimator']]
        #y_preds = cross_val_predict(regr, X=rf_input, cv=cv)
        return np.mean(np.vstack(y_preds), axis=0)

    def fitness_func_multiobjective(ga, solution, solution_idx):
        # decompose the solution into composition part and categorical part
        solution_str = enc_ordinal.inverse_transform(
            [solution[:num_cat_cols]])
        solution_str_df = pd.DataFrame(
            solution_str, columns=cols_type_dict_categories.keys())
        categorical_vec = enc_cat.transform(solution_str_df)
        composition_vec = solution[num_cat_cols:]

        rf_input = np.concatenate(
            (categorical_vec[0], composition_vec), axis=0)
        # we get k predictions as we used k-fold cross validation earlier
        y_preds = [estimator.predict([rf_input]) for estimator in scores['estimator']]
        comp_norm = np.linalg.norm(composition_vec, ord=0)
        #comp_fitness = 1/(comp_norm + 0.000001)
        comp_fitness = -comp_norm
        return (np.mean(y_preds), comp_fitness)

    ga_instance = pygad.GA(
        num_generations=1000,
        num_parents_mating=20,
        mutation_by_replacement=True,
        fitness_func=fitness_func_multiobjective,
        fitness_batch_size=None,
        sol_per_pop=50,
        num_genes=len(value_space),
        gene_space=value_space,
        parent_selection_type='nsga2'
    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = ", solution_fitness)
    print("Index of the best solution : ", solution_idx)
    solution_ord = solution[:num_cat_cols]
    solution_comp = solution[num_cat_cols:]
    solution_str = enc_ordinal.inverse_transform([solution_ord])
    print("Composition solution: ", solution_comp)
    print("Categorical solution: ", solution_str)
    ga_instance.plot_fitness()
    plt.show()





if __name__ == '__main__':
    app.run(main)
