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
    get_value_space,
    filter_singlelayer,
    filter_valid_ratio,
    filter_common,
    filter_compositions,
    CompositionEncoder_DF
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "file", "example_data/psc_all_mz_20240109.csv", "Name of file to read.")


def main(argv):
    df = pd.read_csv(FLAGS.file, index_col=0, low_memory=False)
 
    ### filter dataset for composition and convert composition to norm'd ratio
    composition_dict = {
        "Perovskite_composition_a_ions": "Perovskite_composition_a_ions_coefficients",
        "Perovskite_composition_b_ions": "Perovskite_composition_b_ions_coefficients",
        "Perovskite_composition_c_ions": "Perovskite_composition_c_ions_coefficients",
    }
    df = df.dropna(
        subset=list(composition_dict.keys())+list(composition_dict.values()))
    # filter out multilayer solar cells
    df = filter_singlelayer(df, list(composition_dict.keys()))
    df = filter_valid_ratio(df, list(composition_dict.values()))

    print("Nrows before filtering: ", len(df))
    filter_dict = {
        "Perovskite_composition_a_ions": ["Cs", "MA", "FA"],
        "Perovskite_composition_b_ions": ["Pb"],
        "Perovskite_composition_c_ions": ["I", "Br"],
    }
    ions = []
    for ions_site in filter_dict.values():
        ions += ions_site
    df_filtered = filter_compositions(df, filter_dict)
    print("Nrows after filtering compositions: ", len(df_filtered))

    enc_comp = CompositionEncoder_DF(composition_dict)
    df_comp = enc_comp.fit_transform(df_filtered, append=True)
    df_comp = df_comp.dropna(subset=ions)
    df_comp = df_comp[df_comp["Perovskite_composition_short_form"] != "MAPbI"]
    print("Nrows after dropping na/MAPbI columns: ", len(df_comp))

    # filter common values in categories
    cols_category = [
        "Cell_architecture", "HTL_stack_sequence",
        "Substrate_stack_sequence", "ETL_stack_sequence",
        "Backcontact_stack_sequence",]
    df_common = filter_common(df_comp, cols_category, 0.9)
    print("Nrows after filtering categorical values: ", len(df_common))

    # encode categories with ordinal numbers
    enc_ordinal = OrdinalEncoder()
    X_ord = enc_ordinal.fit_transform(df_common[cols_category])
    for i, col in enumerate(cols_category):
        df_common.loc[:, col+"_ordinal"] = X_ord[:, i]

    ### fit random forest model
    target = "JV_default_PCE"
    cols_composition = ions

    regr = RandomForestRegressor(max_depth=100, random_state=0,
            max_features='sqrt', oob_score=True, n_jobs=-1)

    df_fit = df_common.dropna(subset=target)
    y = df_fit[target]
    enc_cat = TargetEncoder()
    X_cat = enc_cat.fit_transform(df_fit[cols_category], y)
    X_comp = df_fit[cols_composition].to_numpy()
    X_combined = np.concatenate([X_cat, X_comp], axis=1)

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(regr, X=X_combined, y=y, cv=cv,
        scoring=['r2', 'neg_mean_absolute_error'], n_jobs=-1,
        return_estimator=True, error_score='raise')
    print("Scoring attributes: ", scores.keys())
    print("r^2: ", np.mean(scores['test_r2']), "+-", np.std(scores['test_r2']))
    maes = -1*scores['test_neg_mean_absolute_error']
    print("MAE: ", np.mean(maes), "+-", np.std(maes))

    # collect the types of different columns to get the gene space
    cols_ordinal = [col+"_ordinal" for col in cols_category]
    cols_type_dict = {
        **{col: "category" for col in cols_ordinal},
        **{col: "float" for col in cols_composition},
    }
    value_space = get_value_space(df_fit, cols_type_dict)
    #print(value_space)
    num_cat_cols = len(cols_category)

    def fitness_func_batch(ga, solution, solution_idx):
        # decompose the solution into composition part and categorical part
        solution_str = enc_ordinal.inverse_transform(
            solution[:, :num_cat_cols])
        solution_str_df = pd.DataFrame(
            solution_str, columns=cols_category)
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
        num_generations=100,
        num_parents_mating=20,
        mutation_by_replacement=True,
        fitness_func=fitness_func_batch,
        fitness_batch_size=20,
        sol_per_pop=50,
        num_genes=len(value_space),
        gene_space=value_space,
        #parent_selection_type='nsga2'
    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = ", solution_fitness)
    print("Index of the best solution : ", solution_idx)
    solution_ord = solution[:num_cat_cols]
    solution_comp = solution[num_cat_cols:]
    solution_str = enc_ordinal.inverse_transform([solution_ord])
    
    print("Composition solution: ")
    solution_comp_dict = {
        ion: ratio for ion, ratio in zip(cols_composition, solution_comp)}
    print(solution_comp_dict)
    print("Categorical solution: ")
    solution_cat_dict = {
        col: val for col, val in zip(cols_category, solution_str)}
    print(solution_cat_dict)
    ga_instance.plot_fitness()
    plt.show()





if __name__ == '__main__':
    app.run(main)
