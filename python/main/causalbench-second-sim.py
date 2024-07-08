# %%
import os
from functools import partial

import matplotlib.pyplot as plt
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
from src.utils.utils import (
    add_interfix_to_filename,
    create_timestamped_folder,
    get_current_timestamp,
)
from xgboost.sklearn import XGBRegressor

# Purpose: understand impact of SEED on QUANTILE_THRES using linear regression
# Constants:
INPUT_DATA = "../data/processed/genes.csv"
QUANTILE_THRES = 0.025
RESULT_DIR = "../results/output_data/"
RESULT_NAME = "causalbench-first-sim-res.csv"
DATA_NAME = "causalbench-first-sim-data.csv"
N_OBS_SUBSAMPLED = 1000
N_TOP_PREDS = 5
N_TOP_ENVS = 5
PRED_SELECTOR = partial(
    ds.select_top_predictors, n_top_pred=N_TOP_PREDS, environment_column="Z"
)

SEED = 123420  # from https://www.random.org/integers

BCF_0 = BCF(
    n_exog=0,  # needs to know Z
    continuous_mask=np.repeat(True, 0),  # needs to know X
    fx=LinearRegression(),
    gv=LinearRegression(),
    fx_imp=LinearRegression(),
    passes=2,
)

OLS_0 = OLS(fx=LinearRegression())


# Derived constants:
algorithms = [
    (
        "BCF",
        ModelWrapper(BCF_0),
    ),
    ("OLS", ModelWrapper(OLS_0)),
    ("ConstFunc", ModelWrapper(ConstantFunc())),
]

rng = np.random.default_rng(SEED)
timestamp = get_current_timestamp()
result_dir_timestamp = create_timestamped_folder(RESULT_DIR, timestamp)


# %%
# Function definitions
def main():
    # %% import data
    input_data = INPUT_DATA
    data = pd.read_csv(input_data)
    idx_observational = np.where(data[["Z"]] == "non-targeting")[0]
    idx_interventional = np.where(data[["Z"]] != "non-targeting")[0]
    idx_subsample_obs = rng.choice(idx_observational, N_OBS_SUBSAMPLED)
    gene_data = data.iloc[np.concatenate([idx_subsample_obs, idx_interventional])].drop(
        columns=["Z"]
    )
    env_data = data.iloc[np.concatenate([idx_subsample_obs, idx_interventional])][["Z"]]

    results = []
    counter = 0

    # %% for each response variable
    for gene in gene_data.columns[:]:
        counter += 1
        print(f"Processing {counter}/{len(gene_data.columns)}: {gene}")

        all_data = []  # List to collect all dataframes
        # %% for 1 ... N_TOP_ENVS
        for e in range(1, N_TOP_ENVS + 1)[:]:
            # select predictor genes and environment genes
            env_selector = partial(ds.select_top_environments, n_top_env=e)
            preds, envs = ds.select_genes(
                gene, gene_data, env_data, PRED_SELECTOR, env_selector
            )
            X_, y_, Z_ = ds.subset_data(gene, gene_data, env_data, preds, envs)

            # train-test split
            train_test_splitter = partial(
                ds.select_obs_in_training_support, low_quantile=QUANTILE_THRES
            )
            X_train, y_train, Z_train, X_test, y_test, Z_test = ds.test_train_split(
                X_, y_, Z_, "Z", train_test_splitter
            )

            # encode Z
            Z_train_enc = prepare_Z(Z_train)
            Z_test_enc = prepare_Z(Z_test)

            # %% for each algorithm
            for algo_name, algo in algorithms[:]:
                # fit algo
                algo.fit(X_train, y_train.to_numpy().ravel(), Z_train_enc)

                # predict algo on both training and test
                y_train_pred = algo.predict(X_train)
                mse_train = compute_mse(y_train.to_numpy().ravel(), y_train_pred)

                y_test_pred = algo.predict(X_test)
                mse_test = compute_mse(y_test.to_numpy().ravel(), y_test_pred)

                # %% append results
                results.append(
                    {
                        "gene": gene,
                        "n_envs": e,
                        "algorithm": algo_name,
                        "mse_train": mse_train,
                        "mse_test": mse_test,
                    }
                )

                # %% Append identifiers using .loc to avoid SettingWithCopyWarning
                data_length = len(X_train) + len(X_test)
                identifiers = pd.DataFrame(
                    {
                        "set": ["train"] * len(X_train) + ["test"] * len(X_test),
                        "environment": [e] * data_length,
                        "algorithm": [algo_name] * data_length,
                        "y_pred": np.concatenate([y_train_pred, y_test_pred]),
                    }
                )

                combined_df = pd.concat(
                    [
                        pd.concat([y_train, X_train, Z_train], axis=1),
                        pd.concat([y_test, X_test, Z_test], axis=1),
                    ]
                ).reset_index(drop=True)

                # Combine with identifiers
                combined_df_new = pd.concat([combined_df, identifiers], axis=1)
                all_data.append(combined_df_new)

        # %% Concatenate all data for `gene` into a single DataFrame
        final_df = pd.concat(all_data)
        final_df.to_csv(
            os.path.join(
                result_dir_timestamp, add_interfix_to_filename(DATA_NAME, gene)
            ),
            index=False,
        )

    # %% save results to csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(result_dir_timestamp, RESULT_NAME), index=False)


# %%
if __name__ == "__main__":
    print("First gene experiment")
    main()
    print("Done")

# %%
