# %%
import argparse

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import ranksums


# Function definition
def have_same_dist(x: np.ndarray, y: np.ndarray) -> bool:
    return ranksums(x, y)[1] > 0.05


def is_cause(
    gene: int,
    dat_filtered: NDArray[np.float64],
    obs_index: NDArray[np.bool_],
    interventions: NDArray[np.str_],
) -> NDArray[np.int_]:

    # select rows where gene is intervened
    int_index = interventions == gene

    # create observational and interventional data
    dat_obs = dat_filtered[obs_index, :]
    dat_int = dat_filtered[int_index, :]

    # for each gene, check if dat_obs and dat_int have same distribution
    return np.array(
        [
            ~have_same_dist(dat_obs[:, k], dat_int[:, k])
            for k in range(dat_filtered.shape[1])
        ]
    )


# %%
def main(input_path: str, output_path: str, all_rows: bool = False):

    # %%
    # input_path = "../../data/raw/genes/dataset_k562_filtered.npz"
    # Load data
    dat = np.load(input_path)
    # %%
    # Select observational data
    obs_index = dat["interventions"] == "non-targeting"
    obs_data = dat["expression_matrix"][obs_index]

    # Keep only genes that have non-zero expression on observational data
    selected_genes_idx = np.where((obs_data > 0).mean(axis=0) == 1)[0]
    selected_genes_names = dat["var_names"][selected_genes_idx]

    if all_rows:
        row_mask = dat["interventions"] != "excluded"
    else:
        # Rows to extract from dat_filtered and dat["interventions"]
        row_mask = obs_index | np.isin(dat["interventions"], selected_genes_names)
    # %%
    # Datasets only keeping selected_genes_idx
    X = dat["expression_matrix"][row_mask][:, selected_genes_idx]
    Z = dat["interventions"][row_mask]

    dat_df = pd.concat(
        [pd.DataFrame(X, columns=selected_genes_names), pd.DataFrame(Z, columns=["Z"])],
        axis=1,
    )

    # %%
    # output_path = "../../data/processed/genes.csv"
    dat_df.to_csv(output_path, index=False)


# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process gene expression data")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input data file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output data file"
    )
    parser.add_argument(
        "--all_rows",
        required=False,
        action="store_true",
        help="Whether to keep all observations or only the ones corresponding to observational setting + selected genes. Default is False.",
    )

    # how do I pass a boolean argument from terminal?
    args = parser.parse_args()

    main(args.input, args.output, args.all_rows)
