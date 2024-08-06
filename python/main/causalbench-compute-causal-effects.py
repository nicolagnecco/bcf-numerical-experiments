# %%
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from tqdm import tqdm


# Function definition
def compute_causal_effect(
    i: int,
    j: int,
    gene_expressions: NDArray[np.float_],
    gene_expressions_observational: NDArray[np.float_],
    intervention_id: NDArray[np.str_],
    var_names: NDArray[np.str_],
    median: bool = True,
) -> float:
    # select rows where gene is intervened
    int_index = intervention_id == var_names[i]

    # create observational and interventional data
    dat_obs = gene_expressions_observational[:, j]
    dat_int = gene_expressions[int_index, j]

    # return absolute difference of median X_j between observational and interventional data
    if median:
        res = np.abs(np.median(dat_obs, axis=0) - np.median(dat_int, axis=0))
    else:
        res = np.abs(np.mean(dat_obs, axis=0) - np.mean(dat_int, axis=0))
    return res


# %%
def main(input_path: str, output_path: str, median: bool = True):

    # %%
    # input_path = "../data/raw/genes/dataset_k562_filtered.npz"

    # Load data
    dat = np.load(input_path)

    # Select observational data
    obs_index = dat["interventions"] == "non-targeting"
    obs_data = dat["expression_matrix"][obs_index]

    # Pre-compute indices and slices
    gene_expressions = dat["expression_matrix"]
    intervention_id = dat["interventions"]
    var_names = dat["var_names"]
    tot_n_genes = len(var_names)

    def process_pair(i, j) -> Tuple[int, int, float]:
        if i != j:
            return (
                i,
                j,
                compute_causal_effect(
                    i, j, gene_expressions, obs_data, intervention_id, var_names, median
                ),
            )
        return (i, j, 0.0)

    # %%
    # Use joblib to parallelize the double loop
    results = Parallel(n_jobs=-1)(
        delayed(process_pair)(i, j)
        for i in tqdm(range(tot_n_genes))
        for j in range(tot_n_genes)
    )

    # %%

    M = np.zeros((tot_n_genes, tot_n_genes))

    # Place results back into the matrix
    for i, j, value in results:  # type: ignore
        M[i, j] = value

    # Save matrix
    # output_path = "../data/processed/causal-effect-matrix.npy"
    np.save(output_path, M)


# %%
if __name__ == "__main__":

    # input_path = "../data/processed/genes_all.npz"
    # output_path = "../data/processed/causal-effect-matrix.npy"
    # main(input_path, output_path)

    parser = argparse.ArgumentParser(description="Compute causal effect matrix")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input data file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output data file"
    )
    parser.add_argument(
        "--use_median",
        required=False,
        action="store_true",
        help="Whether to compute causal effect using median or mean. Default is False.",
    )
    # how do I pass a boolean argument from terminal?
    args = parser.parse_args()

    main(args.input, args.output, args.use_median)
