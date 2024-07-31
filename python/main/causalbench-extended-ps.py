# %%
import concurrent.futures
import itertools
import os
import shutil
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import configs.genes_extended_parallel_config_ps as cfg
import numpy as np
import pandas as pd
import src.data.data_encoders as de
import src.data.data_selectors as ds
import src.simulations.psweep as ps
from numpy.random import BitGenerator, Generator, SeedSequence
from src.bcf.boosted_control_function_2 import BCF
from src.simulations.simulations_funcs import compute_mse
from src.utils.utils import create_timestamped_folder, get_current_timestamp
from tqdm import tqdm

# %% Constant definitions

DATA = pd.read_csv(cfg.INPUT_DATA)


# %% Function definitions
def main():
    # %% create directory for results
    timestamp = get_current_timestamp()
    result_dir_timestamp = create_timestamped_folder(cfg.RESULT_DIR, timestamp)

    # save config files
    shutil.copy(cfg.INPUT_CONFIG, os.path.join(result_dir_timestamp, cfg.OUTPUT_CONFIG))

    # %% Generate tasks
    tasks = generate_tasks(
        p=cfg.P,
        c=cfg.C,
        r=cfg.R,
        n_iter=cfg.ITERATIONS,
        pred_selector=cfg.PRED_SELECTOR,
        env_selector=cfg.ENV_SELECTOR,
        seed=cfg.SEED,
    )

    sim_wrapper = partial(process_gene_environment_wrapper, algorithms=cfg.algorithms)

    # run simulations in parallel
    ps.run_local(
        sim_wrapper,
        tasks[: cfg.TEST_RUN],
        database_dir=result_dir_timestamp,
        calc_dir=result_dir_timestamp,
        database_basename="database-res.pk",
        simulate=cfg.DRYRUN,
        backup=False,
        skip_dups=False,
        poolsize=cfg.POOLSIZE,
    )
    # save to csv
    df = ps.df_read(os.path.join(f"{result_dir_timestamp}", "database-res.pk"))
    df.to_csv(os.path.join(result_dir_timestamp, cfg.RESULT_NAME), index=False)


def generate_tasks(
    p: int,
    c: int,
    r: int,
    n_iter: int,
    pred_selector: Callable[[str, pd.DataFrame, pd.DataFrame, int], List[str]],
    env_selector: Callable[[List[str], int], List[List[str]]],
    seed: Optional[int] = None,
) -> List[Dict]:

    gene_data = DATA.drop(columns=["Z"])
    environment_data = DATA[["Z"]]

    tasks = []

    # for each response gene generate:
    # - list of preds
    # - list of confounders
    # - list of training envs
    for response_gene in gene_data.columns:
        predictors = pred_selector(response_gene, gene_data, environment_data, p)

        confounders = predictors[:c]

        all_genes = set(gene_data.columns.values)
        all_genes.remove(response_gene)
        remaining_genes = list(all_genes - set(predictors))
        training_environment_combinations = env_selector(remaining_genes, r)

        for training_environments in training_environment_combinations:
            tasks.append(
                {
                    "response_gene": response_gene,
                    "predictors": predictors,
                    "confounders": confounders,
                    "training_environments": training_environments,
                }
            )

    # replicate tasks
    ss = np.random.SeedSequence(seed)
    child_states = ss.spawn(len(tasks) * n_iter)
    new_tasks = []
    for task, child_state in zip(tasks * n_iter, child_states):
        new_tasks.append(task | {"seed": child_state})

    return new_tasks


def flatten_results(nested_results):
    return [item for sublist in nested_results for item in sublist]


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


def process_gene_environment_wrapper(task, algorithms):

    process_gene_environment_algos = partial(
        process_gene_environment, algorithms=algorithms
    )

    # Define the valid keys for the process_gene_environment function
    valid_keys = {
        "response_gene",
        "predictors",
        "confounders",
        "training_environments",
        "seed",
    }

    # Filter out unwanted keys from the task dictionary
    filtered_task = {k: v for k, v in task.items() if k in valid_keys}

    # Call the process_gene_environment function with filtered arguments
    return {"res": process_gene_environment_algos(**filtered_task)}


def process_gene_environment(
    response_gene: str,
    predictors: List[str],
    confounders: List[str],
    training_environments: List[str],
    algorithms,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> List:

    rng = np.random.default_rng(seed)

    # Subsample observational data
    gene_data, env_data = subsample_observational_data(
        num_obs_subsamples=cfg.N_OBS_SUBSAMPLED, seed=rng
    )

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
    # !!!

    # Get list of test data [(X_tests, y_tests, Z_tests), ...]
    all_genes = set(gene_data.columns.values)
    used_genes = set([response_gene] + predictors + training_environments)
    test_environments = list(all_genes - used_genes)

    list_test_data = {
        test_env: ds.subset_data(
            response_gene, gene_data, env_data, predictors, test_env
        )
        for test_env in test_environments
    }

    # Fit and evaluate algos
    results = []
    for algo_name, algo in algorithms:

        # Fit on train
        algo.fit(X_train.to_numpy(), y_train.to_numpy().ravel(), Z_train_enc)

        # !!! for debugging only
        if isinstance(algo.model, BCF):
            M_0 = algo.model.M_0_
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
        )
        results.append(train_results)

        # Evaluate on test
        for test_env in test_environments:
            X_test, y_test, Z_test = list_test_data[test_env]

            test_results = evaluate_model(
                algo=algo,
                X=X_test,
                y=y_test,
                response_gene=response_gene,
                predictors=predictors,
                training_environments=training_environments,
                confounders=confounders,
                test_env=test_env,
                algo_name=algo_name,
                M_0=M_0,
            )

            results.append(test_results)

    return results


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
    }


# %%


if __name__ == "__main__":
    main()
# %%
