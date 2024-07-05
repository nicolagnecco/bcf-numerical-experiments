# %%
from functools import partial
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import pearsonr


def subset_data(
    response_gene: str,
    X: pd.DataFrame,
    Z: pd.DataFrame,
    predictor_genes: List[str],
    environment_genes: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Subsets the data based on the specified genes.

    Parameters
    ----------
    response_gene : str
        Column name of the gene to be used as the response variable Y.
    X : pandas.DataFrame
        DataFrame containing the dataset with genes as columns.
    Z : pandas.DataFrame
        DataFrame containing the dataset with the environment column.
    predictor_genes : list of str
        List of column names for the genes to be used as predictor variables X.
    environment_genes : list of str
        List of gene names used for environmental data subsetting.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing three DataFrames: X (predictors), Y (response), and Z (environment variable).
    """

    # Filter rows according to the specified indices
    row_mask = np.isin(Z, environment_genes)

    # Extract columns for Y, X, and Z
    Y = X[row_mask][[response_gene]]  # Keeping Y as a DataFrame
    X = X[row_mask][predictor_genes]
    Z = Z[row_mask]

    return X, Y, Z


def select_genes(
    response_gene: str,
    X: pd.DataFrame,
    Z: pd.DataFrame,
    select_predictor_fn: Callable[[str, pd.DataFrame, pd.DataFrame], List[str]],
    select_environment_fn: Callable[
        [str, pd.DataFrame, pd.DataFrame, List[str]], List[str]
    ],
) -> Tuple[List[str], List[str]]:
    """
    Selects predictor and environment genes based on a given response gene using provided selection functions.

    Parameters
    ----------
    response_gene : str
        The gene to be used as the response variable.
    X : pandas.DataFrame
        DataFrame containing the dataset with genes as columns.
    Z : pandas.DataFrame
        DataFrame containing the dataset with the environment column.
    select_predictor_fn : function
        A function that takes the DataFrame and the response gene, and returns a list of predictor genes.
    select_environment_fn : function
        A function that takes the DataFrame and the response gene, and returns the name of the environment gene.

    Returns
    -------
    tuple
        A tuple containing lists of predictor genes and the environment gene.
    """

    predictor_genes = select_predictor_fn(response_gene, X, Z)
    environment_genes = select_environment_fn(response_gene, X, Z, predictor_genes)

    return predictor_genes, environment_genes


# placeholder selection functions
def default_predictor_selection(
    response_gene: str, X: pd.DataFrame, Z: pd.DataFrame
) -> List[str]:
    """
    Selects predictor genes; this is a placeholder function.
    """
    # Example: Select all genes except the response variable
    predictors = [col for col in X.columns if col != response_gene]

    return predictors


def select_top_predictors(
    response_gene: str,
    X: pd.DataFrame,
    Z: pd.DataFrame,
    n_top_pred: int,
    environment_column: str,
) -> List[str]:
    """
    Selects predictor genes based on their correlation with the response variable.

    Parameters:
    ----------
    response_gene : str
        The gene to be used as the response variable.
    X : pd.DataFrame
        DataFrame containing the dataset with genes as columns.
    Z : pd.DataFrame
        DataFrame containing the dataset with the environment column. This is not used in this function.
    n_top_pred: int
        The number of top predictors to select
    environment_column: str
        Name of the environment column in `Z`

    Returns:
    -------
    List[str]
        List of predictor genes that are most correlated with the response variable in the observational data.
    """
    # Extract the response variable data
    row_mask = Z[environment_column] == "non-targeting"
    y = X[row_mask][response_gene]

    # Calculate the correlation of each predictor with the response variable
    correlations = {}
    for predictor in X.columns:
        if predictor != response_gene:
            correlation, _ = pearsonr(X[row_mask][predictor], y)
            correlations[predictor] = abs(
                correlation
            )  # Use absolute value to consider both positive and negative correlations

    # Select the top predictors with the highest absolute correlations
    top_predictors = sorted(
        correlations, key=lambda pred: correlations[pred], reverse=True
    )[:n_top_pred]

    return top_predictors


def default_environment_selection(
    response_gene: str, X: pd.DataFrame, Z: pd.DataFrame, preds: List[str]
) -> List[str]:
    """
    Selects environment genes; this is a placeholder function.
    """

    # Example: Select first preds as environment variables + observational data
    environment_genes = preds[:1]
    return environment_genes + ["non-targeting"]


def all_environment_selection(
    response_gene: str, X: pd.DataFrame, Z: pd.DataFrame, preds: List[str]
) -> List[str]:
    """
    Selects as environment all predictor genes + "non-targeting" environment
    """

    # Example: Select all preds as environment variables + observational data
    return preds + ["non-targeting"]


def select_top_environments(
    response_gene: str,
    X: pd.DataFrame,
    Z: pd.DataFrame,
    preds: List[str],
    n_top_env: int,
) -> List[str]:
    """
    Selects as environment the first `n_top_env` in `preds` + "non-targeting" environment
    """

    # Example: Select all preds as environment variables + observational data
    return preds[:n_top_env] + ["non-targeting"]


def test_train_split(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    Z: pd.DataFrame,
    environment_column: str,
    select_train_obs_fn: Callable[
        [pd.DataFrame, pd.DataFrame, pd.Series], NDArray[np.bool_]
    ],
):

    train_mask = select_train_obs_fn(X, Y, Z[environment_column])

    X_train = X[train_mask]
    Y_train = Y[train_mask]
    Z_train = Z[train_mask]

    X_test = X[~train_mask]
    Y_test = Y[~train_mask]
    Z_test = Z[~train_mask]
    return X_train, Y_train, Z_train, X_test, Y_test, Z_test


def default_train_obs_selector(X, Y, Z) -> NDArray[np.bool_]:
    """
    Selects training observations; this is a placeholder function.
    """

    # Example: Select all rows from the observational setting, i.e., "non-targeting"
    row_mask = Z == "non-targeting"
    return row_mask


def select_obs_but_in(X, Y, Z, env_name) -> NDArray[np.bool_]:
    """
    Selects training observations that are not in `env_name`
    """

    # Example: Select all rows from the observational setting, i.e., "non-targeting"
    row_mask = Z != env_name
    return row_mask


def select_obs_in_training_support(X, Y, Z) -> NDArray[np.bool_]:
    """
    Selects training observations in the support of "non-targeting" data
    """

    # Select "non-targeting" observations
    non_targeting_mask = (Z == "non-targeting").values
    non_targeting_X = X[non_targeting_mask]

    # Compute range of X in "non-targeting" environment
    min_values = non_targeting_X.min()
    max_values = non_targeting_X.max()

    # Initialize training mask to include all "non-targeting" obs
    train_mask = non_targeting_mask.copy()

    # Create a combined mask for range condition across all columns
    range_mask = np.ones(len(X), dtype=bool)
    for column in X.columns:
        range_mask &= (X[column] >= min_values[column]) & (
            X[column] <= max_values[column]
        )

    # Combine the non-targeting mask with the range mask
    train_mask |= range_mask

    # Check each other environment (want each observation to be either non-targeting or in the support of targeting)
    return train_mask


# %%
if __name__ == "__main__":

    # %% testing subset_data
    output_path = "../../data/processed/genes.csv"
    environment_column = "Z"
    response_gene = "ENSG00000122406"
    predictor_genes = ["ENSG00000144713", "ENSG00000174748"]
    environment_genes = ["non-targeting", "ENSG00000174748"]
    data = pd.read_csv(output_path)
    X = data.drop(columns=["Z"])
    Z = data[["Z"]]

    # %%
    X_, Y_, Z_ = subset_data(response_gene, X, Z, predictor_genes, environment_genes)

    # %% testing default selectors
    genes = data.drop(columns=["Z"])
    Z = data[["Z"]]
    preds = default_predictor_selection("ENSG00000122406", X, Z)

    all_environment_selection("ENSG00000122406", genes, Z, preds)

    # %% testing select_top_predictors
    top_preds = select_top_predictors(response_gene, X, Z, 10, "Z")

    # %% testing select_top_env
    select_top_environments(response_gene, X, Z, top_preds, 3)

    # %% testing select_genes
    preds, envs = select_genes(
        "ENSG00000122406",
        X,
        Z,
        default_predictor_selection,
        all_environment_selection,
    )

    X_, Y_, Z_ = subset_data(response_gene, X, Z, preds, envs)

    # %% try smarter env selection ...

    # %% try train-test split

    X_train, Y_train, Z_train, X_test, Y_test, Z_test = test_train_split(
        X_, Y_, Z_, "Z", lambda x, y, z: select_obs_but_in(x, y, z, "ENSG00000144713")
    )

    # %% try train-test split in training support
    X_train, Y_train, Z_train, X_test, Y_test, Z_test = test_train_split(
        X_, Y_, Z_, "Z", select_obs_in_training_support
    )

    # %% try pipeline
    preds, envs = select_genes(
        response_gene,
        X,
        Z,
        partial(select_top_predictors, n_top_pred=10, environment_column="Z"),
        partial(select_top_environments, n_top_env=0),
    )

    X_, y_, Z_ = subset_data(response_gene, X, Z, preds, envs)

    X_train, y_train, Z_train, X_test, y_test, Z_test = test_train_split(
        X_, y_, Z_, "Z", select_obs_in_training_support
    )


# %%
