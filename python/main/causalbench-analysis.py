#

import concurrent.futures
import itertools
import os
import shutil
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import configs.causalbench_analysis_config as cfg
import numpy as np
import pandas as pd
import src.data.data_encoders as de
import src.data.data_selectors as ds
from numpy.random import BitGenerator, Generator, SeedSequence
from src.bcf.boosted_control_function_2 import BCF
from src.simulations.simulations_funcs import compute_mse
from src.utils.utils import create_timestamped_folder, get_current_timestamp
from tqdm import tqdm

#  Constant definitions

# import data
DATA = pd.read_csv(cfg.INPUT_DATA)


# Function definitions


def flatten_results(nested_results):
    return [item for sublist in nested_results for item in sublist]


def df_predictions(X, y, Z, algo_name, algo, setting, interv_strength):
    df1 = pd.concat([y, X, Z], axis=1).reset_index(drop=True)

    df2 = pd.DataFrame(
        {
            "algo": algo_name,
            "env": setting,
            "interv_strength": interv_strength,
            "y_pred": algo.predict(X.to_numpy()),
        }
    )

    return pd.concat([df1, df2], axis=1)


def subsample_observational_data(num_obs_subsamples, seed):
    idx_observational = np.where(DATA[["Z"]] == "non-targeting")[0]
    idx_interventional = np.where(DATA[["Z"]] != "non-targeting")[0]
    rng = np.random.default_rng(seed)
    idx_subsample_obs = rng.choice(idx_observational, num_obs_subsamples)

    gene_data = DATA.drop(columns=["Z"]).iloc[
        np.concatenate([idx_subsample_obs, idx_interventional])
    ]
    env_data = DATA[["Z"]].iloc[np.concatenate([idx_subsample_obs, idx_interventional])]

    return gene_data, env_data


def add_confounders(X, y, Z, confounders: List[str]):

    row_mask = Z["Z"] == "non-targeting"

    # add synthetic confounder
    X_conf = X[row_mask][confounders]
    y_mean_train = y[row_mask].values.mean()
    new_y = np.zeros(shape=y.shape)
    new_y[row_mask] = y[row_mask].values

    for conf in confounders:
        new_y[row_mask] = new_y[row_mask] + X_conf[[conf]]

    new_y[~row_mask] = y[~row_mask].values

    new_y[row_mask] = new_y[row_mask] * y_mean_train / new_y[row_mask].mean()

    y_ = pd.DataFrame(
        data=new_y,
        columns=y.columns,
        index=y.index,
    )

    return X, y_, Z


def evaluate_model(
    algo,
    X,
    y,
    response_gene,
    predictors,
    training_environments,
    confounders,
    test_env,
    algo_name,
    M_0,
    interv_strength,
):
    y_pred = algo.predict(X.to_numpy())
    mse = compute_mse(y.to_numpy().ravel(), y_pred)
    return {
        "response": response_gene,
        "predictors": predictors,
        "training_envs": training_environments,
        "confounders": confounders,
        "test_envs": test_env,
        "algorithm": algo_name,
        "mse": mse,
        "M_0": M_0,
        "interv_strength": interv_strength,
    }


def process_gene_environment(
    response_gene: str,
    predictors: List[str],
    training_environments: List[str],
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
    run_id: int = 0,
    iter_id: int = 0,
    debug_dir: str = "./",
) -> List:

    rng = np.random.default_rng(seed)

    # Subsample observational data
    gene_data, env_data = subsample_observational_data(
        num_obs_subsamples=cfg.N_OBS_SUBSAMPLED, seed=rng
    )

    # Get list of test data
    test_environments = predictors

    list_test_data = {
        test_env: ds.subset_data(
            response_gene, gene_data, env_data, predictors, [test_env]
        )
        for test_env in test_environments
    }

    # For each test_env, fit and evaluate algos
    results = []
    df_preds = []
    for test_env in tqdm(
        test_environments, desc="Processing test environments", disable=True
    ):

        # Get X_train, y_train, Z_train
        X_train, y_train, Z_train = ds.subset_data(
            response_gene=response_gene,
            X=gene_data,
            Z=env_data,
            predictor_genes=predictors,
            environment_genes=training_environments,
        )
        Z_train_enc = de.prepare_Z(Z_train)

        # Add confounders
        confounders = None
        if cfg.ADD_CONFOUNDERS:
            confounders = [test_env]
            X_train, y_train, Z_train = add_confounders(
                X_train, y_train, Z_train, confounders
            )

        for algo_name, algo_factory in cfg.algorithms:

            algo = algo_factory()

            # Fit on train
            algo.fit(X_train.to_numpy(), y_train.to_numpy().ravel(), Z_train_enc)

            # !!! for debugging only
            if cfg.DEBUG_PREDICTIONS:
                if isinstance(algo.model, BCF):
                    M_0 = algo.model.M_0_
                else:
                    M_0 = None
            else:
                M_0 = None

            # Evaluate on train
            train_results = evaluate_model(
                algo=algo,
                X=X_train,
                y=y_train,
                response_gene=response_gene,
                predictors=predictors,
                training_environments=training_environments,
                confounders=confounders,
                test_env="train",
                algo_name=algo_name,
                M_0=M_0,
                interv_strength=0,
            )
            train_results["run_id"] = run_id
            train_results["iter_id"] = iter_id
            results.append(train_results)

            if cfg.DEBUG_PREDICTIONS:
                df_pred = df_predictions(
                    X=X_train,
                    y=y_train,
                    Z=Z_train,
                    algo_name=algo_name,
                    algo=algo,
                    setting="train",
                    interv_strength=0,
                )
                df_preds.append(df_pred)

            X_test, y_test, Z_test = list_test_data[test_env]

            for perc in cfg.TEST_PERCENTAGES:

                test_mask = ds.get_test_mask_perc(
                    X_test,
                    Z_test,
                    [test_env],
                    100 * (1 - perc),
                    100 * (1),
                )

                test_results = evaluate_model(
                    algo=algo,
                    X=X_test[test_mask],
                    y=y_test[test_mask],
                    response_gene=response_gene,
                    predictors=predictors,
                    training_environments=training_environments,
                    confounders=confounders,
                    test_env=test_env,
                    algo_name=algo_name,
                    M_0=M_0,
                    interv_strength=perc,
                )
                test_results["run_id"] = run_id
                test_results["iter_id"] = iter_id
                results.append(test_results)

                if cfg.DEBUG_PREDICTIONS:
                    df_pred = df_predictions(
                        X=X_test[test_mask],
                        y=y_test[test_mask],
                        Z=Z_test[test_mask],
                        algo_name=algo_name,
                        algo=algo,
                        setting="test",
                        interv_strength=perc,
                    )
                    df_preds.append(df_pred)

    if cfg.DEBUG_PREDICTIONS:
        final_df = pd.concat(df_preds)
        final_df.to_csv(
            os.path.join(debug_dir, f"debug_response_{response_gene}.csv"),
            index=False,
        )

    return results


def generate_tasks(
    p: int,
    r: int,
    n_iter: int,
    pred_selector: Callable[[str, pd.DataFrame, pd.DataFrame, int], List[str]],
    select_candidate_envs: Callable[[List[str], str, List[str], List[str]], List[str]],
    env_selector: Callable[[List[str], int], List[List[str]]],
    seed: Optional[int] = None,
) -> List[Tuple[str, List[str], List[str], SeedSequence, int, int]]:

    gene_data = DATA.drop(columns=["Z"])
    environment_data = DATA[["Z"]]

    tasks = []

    # for each response gene generate:
    # - list of preds
    # - list of confounders
    # - list of training envs
    for response_gene in gene_data.columns:
        predictors = pred_selector(response_gene, gene_data, environment_data, p)

        task_response_preds = [(response_gene, predictors)]

        all_genes = np.unique(environment_data).tolist()

        remaining_genes = select_candidate_envs(
            all_genes, response_gene, predictors, predictors
        )

        training_environment_combinations = env_selector(remaining_genes, r)

        task_reponse_preds_conf_trenv = [
            (*a, b)
            for a in task_response_preds
            for b in training_environment_combinations
        ]

        tasks.extend(task_reponse_preds_conf_trenv)

    # replicate tasks
    ss = np.random.SeedSequence(seed)
    child_states = ss.spawn(len(tasks) * n_iter)
    new_tasks = []
    run_id = 0
    for task, child_state in zip(tasks * n_iter, child_states):
        n_iter_id = run_id // len(tasks)
        new_tasks.append((*task, child_state, run_id, n_iter_id))
        run_id += 1

    return new_tasks


def main():

    # create directory for results and debug
    timestamp = get_current_timestamp()
    result_dir_timestamp = create_timestamped_folder(cfg.RESULT_DIR, timestamp)
    debug_dir_timestamp = create_timestamped_folder(result_dir_timestamp, "_debug")

    # save config files
    shutil.copy(cfg.INPUT_CONFIG, os.path.join(result_dir_timestamp, cfg.OUTPUT_CONFIG))

    # Generate tasks
    tasks = generate_tasks(
        p=cfg.P,
        r=cfg.R,
        n_iter=cfg.ITERATIONS,
        pred_selector=cfg.PRED_SELECTOR,
        select_candidate_envs=cfg.CANDIDATE_ENV_SELECTOR,
        env_selector=cfg.ENV_SELECTOR,
        seed=cfg.SEED,
    )

    #  Run tasks
    results = []

    if cfg.SEQUENTIAL:
        # Sequential mode for debugging
        for (
            response_gene,
            predictors,
            training_envs,
            seed,
            run_id,
            iter_id,
        ) in tqdm(tasks[cfg.FIRST_TASK : cfg.LAST_TASK], desc="Processing tasks"):
            result = process_gene_environment(
                response_gene,
                predictors,
                training_envs,
                seed,
                run_id,
                iter_id,
                debug_dir_timestamp,
            )

            # for res in result:
            # res["run_id"] = run_id

            results.append(result)
    else:
        # Parallel mode
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_gene_environment,
                    response_gene,
                    predictors,
                    training_envs,
                    seed,
                    run_id,
                    iter_id,
                    debug_dir_timestamp,
                )
                for response_gene, predictors, training_envs, seed, run_id, iter_id in tasks[
                    cfg.FIRST_TASK : cfg.LAST_TASK
                ]
            ]

            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                result = future.result()

                # for res in result:
                # res["run_id"] = run_id

                results.append(result)

    # % Flatten results
    flattened_results = flatten_results(results)

    # Convert to DataFrame
    results_df = pd.DataFrame(flattened_results)

    # Save results to csv or parquet
    if cfg.USE_NPZ:
        npz_path = os.path.join(
            result_dir_timestamp, cfg.RESULT_NAME.replace(".csv", ".npz")
        )

        # Save each column as a separate array
        np.savez_compressed(
            npz_path, **{col: results_df[col].values for col in results_df.columns}
        )
    else:
        results_df.to_csv(
            os.path.join(result_dir_timestamp, cfg.RESULT_NAME), index=False
        )


if __name__ == "__main__":
    main()
