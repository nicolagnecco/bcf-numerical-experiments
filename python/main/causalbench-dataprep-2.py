# %%
import argparse

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import ranksums


# Function definition
# %%
def main(
    gene_path: str,
    matrix_path: str,
    output_path: str,
    n_genes: int = 10,
    all_rows: bool = False,
):

    # %%
    # Load data
    # gene_path = "../data/processed/genes_all.npz"
    dat = np.load(gene_path)
    obs_index = dat["interventions"] == "non-targeting"

    # Load causal effects
    # matrix_path = "../data/processed/causal-effect-matrix-2.npy"
    M = np.load(matrix_path)

    # %%
    # Recursevly remove genes whose rowmax is smallest until M has n_genes rows and columns
    genes_names = dat["var_names"]
    while M.shape[0] > n_genes:
        rowmax = M.max(axis=1)
        idx = np.argmin(rowmax)
        M = np.delete(M, idx, axis=0)
        M = np.delete(M, idx, axis=1)
        genes_names = np.delete(genes_names, idx)

    # %%
    # Keep genes in genes_names
    selected_genes_names = genes_names
    selected_genes_idx = np.where(np.isin(dat["var_names"], selected_genes_names))[0]

    # %%
    # Keep rows
    if all_rows:
        row_mask = dat["interventions"] != "excluded"
    else:
        row_mask = obs_index | np.isin(dat["interventions"], selected_genes_names)

    # Datasets only keeping selected_genes_idx
    X = dat["expression_matrix"][row_mask][:, selected_genes_idx]
    Z = dat["interventions"][row_mask]

    dat_df = pd.concat(
        [pd.DataFrame(X, columns=selected_genes_names), pd.DataFrame(Z, columns=["Z"])],
        axis=1,
    )

    # %%
    # Save data
    # output_path = "../data/processed/genes_causal_effect.csv"
    dat_df.to_csv(output_path, index=False)


# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process gene expression data")
    parser.add_argument(
        "--gene_path", type=str, required=True, help="Path to input genes file"
    )
    parser.add_argument(
        "--matrix_path",
        type=str,
        required=True,
        help="Path to input causla effect matrix file",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output data file"
    )
    parser.add_argument(
        "--n_genes", type=int, required=True, help="Number of genes to select"
    )
    parser.add_argument(
        "--all_rows",
        required=False,
        action="store_true",
        help="Whether to keep all observations or only the ones corresponding to observational setting + selected genes. Default is False.",
    )

    # how do I pass a boolean argument from terminal?
    args = parser.parse_args()

    main(args.gene_path, args.matrix_path, args.output, args.n_genes, args.all_rows)
