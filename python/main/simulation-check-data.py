# %%
import os
import shutil
from functools import partial

import configs.check_data_config as cfg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.data.data_selectors as ds
from scipy.stats import pearsonr
from src.data.data_encoders import prepare_Z
from src.scenarios.generate_data import generate_data_Z_Gaussian
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
    results = []

    # %% for each response variable
    for i in range(cfg.n_sims):
        # %%
        print(f"Processing iteration {i+1}/{cfg.n_sims}")

        all_data = []  # List to collect all dataframes

        (
            X_train,
            X_test,
            y_train,
            y_test,
            Z_train,
            Z_test,
            f_train,
            f_test,
            f_tree,
            M,
            S,
            gamma,
        ) = generate_data_Z_Gaussian(
            n=cfg.n,
            p=cfg.p,
            p_effective=cfg.p_effective,
            tree_depth=cfg.tree_depth,
            r=cfg.r,
            interv_strength=cfg.interv_strength,
            gamma_norm=cfg.gamma_norm,
            sd_y=cfg.sd_y,
            seed=rng,
        )
        # encode Z
        Z_train_enc = prepare_Z(Z_train)
        Z_test_enc = prepare_Z(Z_test)

        # %% for each algorithm
        for algo_name, algo in cfg.algorithms[:]:
            # fit algo
            algo.fit(X_train, y_train.ravel(), Z_train_enc)

            # predict algo on both training and test
            y_train_pred = algo.predict(X_train)
            mse_train = compute_mse(y_train.ravel(), y_train_pred)

            y_test_pred = algo.predict(X_test)
            mse_test = compute_mse(y_test.ravel(), y_test_pred)

            # %% append results
            p = X_train.shape[1]
            results.append(
                {
                    "iter": i,
                    "algorithm": algo_name,
                    "mse_train": mse_train,
                    "mse_test": mse_test,
                }
            )

            # %% Append identifiers using .loc to avoid SettingWithCopyWarning
            data_length = len(X_train) + len(X_test)
            combined_df = pd.DataFrame(
                {
                    "set": ["train"] * len(X_train) + ["test"] * len(X_test),
                    "algorithm": [algo_name] * data_length,
                    "y_pred": np.concatenate([y_train_pred, y_test_pred]),
                    "Z": np.concatenate([Z_train[:, 0], Z_test[:, 0]]),
                    "X1": np.concatenate([X_train[:, 0], X_test[:, 0]]),
                    "X2": np.concatenate([X_train[:, 1], X_test[:, 1]]),
                    "y": np.concatenate([y_train, y_test]),
                }
            )

            all_data.append(combined_df)

        # %% Concatenate all data for `gene` into a single DataFrame
        final_df = pd.concat(all_data)
        final_df.to_csv(
            os.path.join(
                result_dir_timestamp, add_interfix_to_filename(cfg.DATA_NAME, i)
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
