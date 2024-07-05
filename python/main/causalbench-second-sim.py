# %%
from functools import partial

import numpy as np
import pandas as pd
import src.data.data_selectors as ds
import src.simulations.psweep as ps
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.algorithms.algorithm_wrapper import ModelWrapper
from src.algorithms.oracle_methods import ConstantFunc
from src.bcf.boosted_control_function_2 import BCF, OLS
from src.data.data_encoders import prepare_Z
from src.simulations.parameter_grids import full_grid
from src.simulations.simulations_funcs import compute_mse, run_simulation
from xgboost.sklearn import XGBRegressor

# Constants:
RESULT_NAME = "../results/output_data/causalbench-first-sim-res-linreg-newtopcor.csv"
DATA_NAME = "../results/output_data/causalbench-first-sim-data.csv"
N_TOP_PREDS = 5
N_TOP_ENVS = 5
PRED_SELECTOR = partial(
    ds.select_top_predictors, n_top_pred=N_TOP_PREDS, environment_column="Z"
)

SEED = 485626181  # from https://www.random.org/integers

# different algorithms
# - all rf?

# different data selection
# - keep only int obs that are well within support?
# - sample subset of obs data for training?
# -

# different dataset
# - try dataset from TPAMI paper


bcf = BCF(
    n_exog=0,  # needs to know Z
    continuous_mask=np.repeat(True, 0),  # needs to know X
    fx=LinearRegression(),  # RandomForestRegressor(n_estimators=20),  # DecisionTreeRegressor(),
    gv=LinearRegression(),  # RandomForestRegressor(n_estimators=20),  # DecisionTreeRegressor(),
    fx_imp=LinearRegression(),  # RandomForestRegressor(
    #  # DecisionTreeRegressor(),
    passes=2,
)

ols = OLS(fx=LinearRegression())

ALGORITHMS = [
    (
        "BCF",
        ModelWrapper(bcf),
    ),
    ("OLS", ModelWrapper(ols)),
    ("ConstFunc", ModelWrapper(ConstantFunc())),
]


# %%
# Function definitions
def main():
    # 1. DATA
    # import data
    input_data = "../data/processed/genes.csv"
    data = pd.read_csv(input_data)
    gene_data = data.drop(columns=["Z"])
    env_data = data[["Z"]]

    results = []
    counter = 0

    # for each response variable
    for gene in gene_data.columns[:]:
        counter += 1
        print(f"Processing {counter}/{len(gene_data.columns)}: {gene}")

        # for 1 ... N_TOP_ENVS
        for e in range(1, N_TOP_ENVS + 1):
            # select predictor genes and environment genes
            env_selector = partial(ds.select_top_environments, n_top_env=e)
            preds, envs = ds.select_genes(
                gene, gene_data, env_data, PRED_SELECTOR, env_selector
            )

            X_, y_, Z_ = ds.subset_data(gene, gene_data, env_data, preds, envs)

            # for each environment which is != "non-targeting"
            # for env in np.array(envs)[np.array(envs) != "non-targeting"][:1]:
            # train-test split
            X_train, y_train, Z_train, X_test, y_test, Z_test = ds.test_train_split(
                X_, y_, Z_, "Z", ds.select_obs_in_training_support
            )

            # encode Z
            Z_train_enc = prepare_Z(Z_train)
            Z_test_enc = prepare_Z(Z_test)

            # for each algorithm
            for algo_name, algo in ALGORITHMS:
                # fit algo
                algo.fit(X_train, y_train.to_numpy().ravel(), Z_train_enc)

                # predict algo on both training and test
                y_train_pred = algo.predict(X_train)
                mse_train = compute_mse(y_train.to_numpy().ravel(), y_train_pred)

                y_test_pred = algo.predict(X_test)
                mse_test = compute_mse(y_test.to_numpy().ravel(), y_test_pred)

                # append results
                results.append(
                    {
                        "gene": gene,
                        "n_envs": e,
                        "algorithm": algo_name,
                        "mse_train": mse_train,
                        "mse_test": mse_test,
                    }
                )

    # save results to csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULT_NAME, index=False)

    # save data to csv
    data.to_csv(DATA_NAME, index=True)


# %%
if __name__ == "__main__":
    print("First gene experiment")
    main()
    print("Done")

# %%
