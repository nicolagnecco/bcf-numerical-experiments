# %%
import os
import shutil
from functools import partial

import configs.genes_exp_5_config as cfg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.data.data_selectors as ds
from scipy.stats import pearsonr
from src.bcf.boosted_control_function_2 import BCF
from src.data.data_encoders import prepare_Z
from src.simulations.simulations_funcs import compute_mse
from src.utils.utils import (
    add_interfix_to_filename,
    create_timestamped_folder,
    get_current_timestamp,
)

# Derived constants:
rng = np.random.default_rng(cfg.SEED)
timestamp = get_current_timestamp()
result_dir_timestamp = create_timestamped_folder(cfg.RESULT_DIR, timestamp)


# %%
# Function definitions
def main():
    # %% import data
    input_data = cfg.INPUT_DATA
    data = pd.read_csv(input_data)
    idx_observational = np.where(data[["Z"]] == "non-targeting")[0]
    idx_interventional = np.where(data[["Z"]] != "non-targeting")[0]
    idx_subsample_obs = rng.choice(idx_observational, cfg.N_OBS_SUBSAMPLED)
    gene_data_full = data.drop(columns=["Z"])
    env_data_full = data[["Z"]]
    gene_data = gene_data_full.iloc[
        np.concatenate([idx_subsample_obs, idx_interventional])
    ]
    env_data = env_data_full.iloc[
        np.concatenate([idx_subsample_obs, idx_interventional])
    ]

    results = []
    counter = 0

    # %% for each response variable
    for gene in gene_data.columns[cfg.first_gene : cfg.last_gene]:
        counter += 1
        print(f"Processing {counter}/{len(gene_data.columns)}: {gene}")

        all_data = []  # List to collect all dataframes
        # %% for each environment
        for e in range(cfg.N_TOP_PREDS)[:]:
            print(f"Processing environment {e + 1} out of {cfg.N_TOP_PREDS}")
            # %% select predictor genes and environment genes
            env_selector = partial(ds.select_environment_e, e=e)
            preds, envs = ds.select_genes(
                gene, gene_data_full, env_data_full, cfg.PRED_SELECTOR, env_selector
            )

            # subset data
            X_, y_, Z_ = ds.subset_data(gene, gene_data, env_data, preds, envs)

            # %% loop through interv strength
            for n_env_top in [20, 40, 60, 80, 100]:

                # %% train-test split
                n_env_top = int(n_env_top)
                train_test_splitter = partial(
                    ds.select_obs_in_observational_support,
                    low_quantile=cfg.QUANTILE_THRES,
                    n_env_top=n_env_top,
                )
                X_train, y_train, Z_train, X_test, y_test, Z_test = ds.test_train_split(
                    X_, y_, Z_, "Z", train_test_splitter
                )

                # encode Z
                Z_train_enc = prepare_Z(Z_train)
                Z_test_enc = prepare_Z(Z_test)

                # %% for each algorithm
                for algo_name, algo in cfg.algorithms[:]:
                    # fit algo
                    algo.fit(
                        X_train.to_numpy(), y_train.to_numpy().ravel(), Z_train_enc
                    )

                    # predict algo on both training and test
                    y_train_pred = algo.predict(X_train.to_numpy())
                    mse_train = compute_mse(y_train.to_numpy().ravel(), y_train_pred)

                    y_test_pred = algo.predict(X_test.to_numpy())
                    mse_test = compute_mse(y_test.to_numpy().ravel(), y_test_pred)

                    if isinstance(algo.model, BCF):
                        M_0 = algo.model.M_0_
                    else:
                        M_0 = None

                    # %% append results
                    results.append(
                        {
                            "gene": gene,
                            "env_id": e,
                            "n_env_obs": n_env_top,
                            "algorithm": algo_name,
                            "mse_train": mse_train,
                            "mse_test": mse_test,
                            "M_0": M_0,
                        }
                    )

                    # %% Append identifiers using .loc to avoid SettingWithCopyWarning
                    data_length = len(X_train) + len(X_test)
                    identifiers = pd.DataFrame(
                        {
                            "set": ["train"] * len(X_train) + ["test"] * len(X_test),
                            "environment": [e] * data_length,
                            "n_env_top": [n_env_top] * data_length,
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
                result_dir_timestamp, add_interfix_to_filename(cfg.DATA_NAME, gene)
            ),
            index=False,
        )

    # %% save results to csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(result_dir_timestamp, cfg.RESULT_NAME), index=False)

    # save config files
    shutil.copy(cfg.INPUT_CONFIG, os.path.join(result_dir_timestamp, cfg.OUTPUT_CONFIG))


# %%
if __name__ == "__main__":
    print("First gene experiment")
    main()
    print("Done")

# %%