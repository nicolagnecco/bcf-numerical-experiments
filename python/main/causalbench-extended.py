# %%
import concurrent.futures
import itertools
import random
from typing import Callable, List, Optional, Tuple, Union

import configs.genes_exp_parallel_config as cfg
import numpy as np
import pandas as pd


# %%
# Example deterministic function to select p predictors
def select_predictors(j, p, total_genes):
    all_genes = set(range(1, total_genes + 1))
    all_genes.remove(j)
    predictors = random.sample(sorted(all_genes), p)
    return predictors


# Example function to process a gene with a specific environment combination
def process_gene_environment(j, predictors, environment_combination):
    # You can add any specific processing here
    return j, predictors, environment_combination


def env_selector(remaining_genes: List[str], k: int) -> List[Tuple[str, ...]]:
    return list(itertools.combinations(remaining_genes, k))


def generate_tasks(
    gene_data: pd.DataFrame,
    environment_data: pd.DataFrame,
    p: int,
    k: int,
    n_iter: int,
    pred_selector: Callable[[str, pd.DataFrame, pd.DataFrame, int], List[str]],
    env_selector: Callable[[List[str], int], List[Tuple[str, ...]]],
    seed: Optional[int] = None,
) -> List[Tuple[str, List[str], List[str]]]:
    tasks = []

    # generate for each response gene, list of preds + list of training envs
    for response_gene in gene_data.columns:
        predictors = pred_selector(response_gene, gene_data, environment_data, p)
        all_genes = set(gene_data.columns.values)
        all_genes.remove(response_gene)
        remaining_genes = list(all_genes - set(predictors))
        environment_combinations = env_selector(remaining_genes, k)

        for combination in environment_combinations:
            tasks.append((response_gene, predictors, list(combination)))

    # replicate tasks
    ss = np.random.SeedSequence(seed)
    child_states = ss.spawn(len(tasks) * n_iter)
    new_tasks = []
    for task, child_state in zip(tasks * n_iter, child_states):
        new_tasks.append((*task, child_state))

    return new_tasks


def main():
    p = 3  # Number of predictors !!!
    k = 1  # Number of environments !!!
    iterations = 10

    data = pd.read_csv(cfg.INPUT_DATA)
    gene_data_full = data.drop(columns=["Z"])
    env_data_full = data[["Z"]]

    tasks = generate_tasks(
        gene_data=gene_data_full,
        environment_data=env_data_full,
        p=p,
        k=k,
        n_iter=iterations,
        pred_selector=cfg.PRED_SELECTOR,
        env_selector=env_selector,
        seed=cfg.SEED,
    )

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_gene_environment, j, predictors, combination)
            for j, predictors, combination in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Process the results as needed
    for result in results:
        j, predictors, environment_combination = result
        print(
            f"Gene {j}: Predictors {predictors}, Environments {environment_combination}"
        )

    # put subsampling inside process_gene_environment function !!!
    # idx_observational = np.where(data[["Z"]] == "non-targeting")[0]
    # idx_interventional = np.where(data[["Z"]] != "non-targeting")[0]
    # idx_subsample_obs = rng.choice(idx_observational, cfg.N_OBS_SUBSAMPLED)

    # gene_data = gene_data_full.iloc[
    #     np.concatenate([idx_subsample_obs, idx_interventional])
    # ]
    # env_data = env_data_full.iloc[
    #     np.concatenate([idx_subsample_obs, idx_interventional])
    # ]


if __name__ == "__main__":
    main()

# %%
