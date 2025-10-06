from typing import Tuple

import numpy as np


def split_X_and_Z(X: np.ndarray, n_exog: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the concatenated [X, Z] matrix into separate X and Z.

    Parameters:
    -----------
    X : numpy.ndarray
        The concatenated [X, Z] matrix.

    n_exog : int
        Number of exogenous variables.

    Returns:
    --------
    Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the split X and Z matrices.
    """
    # Split `X = [X, Z]` into `X_original` and `Z` knowing the number of cols in `Z`
    X_original = X[:, :-n_exog]
    Z = X[:, -n_exog:]
    return X_original, Z


def split_cont_cat(continuous_mask, X) -> Tuple[np.ndarray, np.ndarray]:
    m = np.asarray(continuous_mask, dtype=bool)
    if m.ndim != 1 or m.size != X.shape[1]:
        raise ValueError("continuous_mask must be 1D bool of length n_cols(X).")

    X_cont = X[:, m]
    X_cat = X[:, ~m]
    return X_cont, X_cat
