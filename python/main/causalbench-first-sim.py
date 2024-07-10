# %%
from functools import partial

import numpy as np
import pandas as pd
import src.data.data_selectors as ds
import src.simulations.psweep as ps
from sklearn.tree import DecisionTreeRegressor
from src.algorithms.algorithm_wrapper import ModelWrapper
from src.algorithms.oracle_methods import ConstantFunc
from src.bcf.boosted_control_function_2 import BCF, OLS
from src.data.data_encoders import prepare_Z
from src.simulations.parameter_grids import full_grid
from src.simulations.simulations_funcs import compute_mse, run_simulation

# Constants:
RESULT_NAME = "../results/output_data/causalbench-first-sim-res.csv"
DATA_NAME = "../results/output_data/causalbench-first-sim-data.csv"


SEED = 485626181  # from https://www.random.org/integers

ALGORITHMS = [
    (
        "BCF",
        ModelWrapper(
            BCF(
                n_exog=0,  # needs to know Z
                continuous_mask=np.repeat(True, 0),  # needs to know X
                fx=DecisionTreeRegressor(),
                gv=DecisionTreeRegressor(),
                fx_imp=DecisionTreeRegressor(),
                passes=2,
            )
        ),
    ),
    ("OLS", ModelWrapper(OLS(fx=DecisionTreeRegressor()))),
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
    for gene in gene_data.columns[:1]:
        counter += 1
        print(f"Processing {counter}/{len(gene_data.columns)}: {gene}")

        # select predictor genes and environment genes
        preds, envs = ds.select_genes(
            gene,
            gene_data,
            env_data,
            ds.default_predictor_selection,
            ds.all_environment_selection,
        )

        X_, y_, Z_ = ds.subset_data(gene, gene_data, env_data, preds, envs)

        # for each environment which is != "non-targeting"
        for env in np.array(envs)[np.array(envs) != "non-targeting"][:1]:
            # train-test split
            X_train, y_train, Z_train, X_test, y_test, Z_test = ds.test_train_split(
                X_, y_, Z_, "Z", lambda x, y, z: ds.select_obs_but_in(x, y, z, env)
            )

            # encode Z
            Z_train_enc = prepare_Z(Z_train)
            Z_test_enc = prepare_Z(Z_test)

            # for each algorithm
            for algo_name, algo in ALGORITHMS:
                # fit algo
                algo.fit(X_train.to_numpy(), y_train.to_numpy().ravel(), Z_train_enc)

                # predict algo on both training and test
                y_train_pred = algo.predict(X_train.to_numpy())
                mse_train = compute_mse(y_train.to_numpy().ravel(), y_train_pred)

                y_test_pred = algo.predict(X_test.to_numpy())
                mse_test = compute_mse(y_test.to_numpy().ravel(), y_test_pred)

                # append results
                results.append(
                    {
                        "gene": gene,
                        "environment": env,
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
