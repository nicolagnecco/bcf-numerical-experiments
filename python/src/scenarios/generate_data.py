from typing import Optional, Tuple, Union

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence
from src.scenarios.generate_x import (
    generate_M,
    generate_orthonormal_vector,
    generate_random_covariance,
    generate_random_mat,
    sample_Gaussian,
    sample_Z_onehot,
    sample_Z_sphere,
)
from src.scenarios.generate_y import generate_tree
from src.utils.data_definitions import Tree


def generate_data(
    n: int,
    p: int,
    p_effective: int,
    tree_depth: int,
    r: int,
    interv_strength: float,
    gamma_norm: float,
    sd_y: float,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tree,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Generate data from a confounded SCM

    Generates training and test data from the following SCM,

        `X = interv_strength * M @ Z + V`,

        `y = f(X) + gamma @ V + N(0, sd_y^2)`,

    where

    - `X` is the predictor vector,
    - `y` is the response,
    - `Z` denotes different environments,
    - `M` is the matrix whose columns are the directions spanned
     by the environments,
    - `interv_strength` is the intervention strength along the columns
     of `M` at test time (at training time is equal to 1),
    - `V` is the vector of confounders that follow a joint Gaussian distribution,
    - `f(X)` is the causal function, i.e., a tree with depth `tree_depth`
     which depends on the first `p_effective` predictors,
     - `gamma` is the vector that controls how much `V` confounds
     the predictors and the response
     - `N(0, sd_y^2)` is the reponse's independent noise term with
     variance equal to `sd_y^2`.


    Parameters
    ----------
    n : int
        The number of observations.

    p : int
        The number of dimensions.

    p_effective: int
        The number of dimensions that affect the response variable.

    tree_depth : int
        The depth of the tree, which controls the complexity of the
        causal function (higher = more complex).

    r : int
        The rank of the matrix `M` (can think of `r` ~ number of environments).

    interv_strength: float > 0
        The strength of the intervention. It controls the perturbation
        of the test observations in the predictor space.

    gamma_norm: float > 0
        The strength of the confounder.

    sd_y: float > 0
        The standard deviation of the response's independent noise term.

    seed : {None, int, SeedSequence, BitGenerator, Generator}, optional
            A seed to initialize the `BitGenerator`. If `None`, then fresh,
            unpredictable entropy will be pulled from the OS. If an `int`
            is passed, then it will be passed to `SeedSequence` to derive
            the initial `BitGenerator` state. One may also pass in a `SeedSequence` instance.
            Additionally, when passed a `BitGenerator`, it will be wrapped by `Generator`.
            If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    tuple, made of
    - X_train, X_test: np.ndarray of predictors,
    - y_train, y_test: np.ndarray of responses,
    - Z_train: np.ndarray of environments,
    - f_train, f_test: np.ndarray of causal function.
    """

    # TODO: define datatype for (X_train, X_test, y_train, y_test, Z_train_cat, f_train, f_test)?
    # TODO: what about instrument strength?

    rng = np.random.default_rng(seed)

    M = generate_M(
        p, r, rng
    )  # TODO: perhaps just random length-1 vectors (instead of orthonormal matrices)?
    S = generate_random_covariance(
        ndims=p, max_condition_number=1, total_variance=0.1, seed=rng
    )
    f_ = generate_tree(
        p_effective, tree_depth, min_leaf_width=0.2, scale_y=0.5, limit_x=0.5, seed=rng
    )
    gamma = generate_orthonormal_vector(ndims=p, seed=rng) * gamma_norm

    V_train = sample_Gaussian(n=n, ndims=p, S=S, seed=rng)
    Z_train, Z_train_cat = sample_Z_onehot(
        n=n, r=r, seed=rng
    )  # TODO: perhaps sample from sphere here as well?
    noise_train = rng.normal(scale=sd_y, size=n)
    X_train = Z_train @ M.T + V_train
    f_train = f_.predict(X_train[:, range(p_effective)])
    y_train = f_train + V_train @ gamma + noise_train

    V_test = sample_Gaussian(n=n, ndims=p, S=S, seed=rng)
    Z_test = sample_Z_sphere(n=n, r=r, seed=rng)
    noise_test = rng.normal(scale=sd_y, size=n)
    X_test = interv_strength * Z_test @ M.T + V_test
    f_test = f_.predict(X_test[:, range(p_effective)])
    y_test = f_test + V_test @ gamma + noise_test

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        Z_train_cat,
        f_train,
        f_test,
        f_,
        M,
        S,
        gamma,
    )


def generate_data_Z_Gaussian(
    n: int,
    p: int,
    p_effective: int,
    tree_depth: int,
    r: int,
    interv_strength: float,
    gamma_norm: float,
    sd_y: float,
    q: Optional[int] = None,
    mean_Z: float = 0.0,
    mean_X: float = 0.0,
    eigengap: float = 1.0,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tree,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Generate data from a confounded SCM

    Generates training and test data from the following SCM,

        `X = M @ (Z * interv_strength) + V`,

        `y = f(X) + gamma @ V + N(0, sd_y^2)`,

    where

    - `X` is the predictor vector,
    - `y` is the response,
    - `Z` is the environment vector that follow a joint Gaussian distribution,
    - `M` is the matrix whose columns are the directions spanned
     by the environments,
    - `interv_strength` is the intervention strength along the columns
     of `M` at test time (at training time is equal to 1),
    - `V` is the vector of confounders that follow a joint Gaussian distribution,
    - `f(X)` is the causal function, i.e., a tree with depth `tree_depth`
     which depends on the first `p_effective` predictors,
     - `gamma` is the vector that controls how much `V` confounds
     the predictors and the response
     - `N(0, sd_y^2)` is the reponse's independent noise term with
     variance equal to `sd_y^2`.


    Parameters
    ----------
    n : int
        The number of observations.

    p : int
        The number of dimensions.

    p_effective: int
        The number of dimensions that affect the response variable.

    tree_depth : int
        The depth of the tree, which controls the complexity of the
        causal function (higher = more complex).

    r : int
        The number of columns of the matrix `M` (can think of `r` ~ number of environments).

    interv_strength: float > 0
        The strength of the intervention. It controls the perturbation
        of the test observations in the predictor space.

    gamma_norm: float > 0
        The strength of the confounder. It is expressed as the norm of the `gamma` vector.

    sd_y: float > 0
        The standard deviation of the response's independent noise term.

     q : int
        The rank of the matrix `M`.

    seed : {None, int, SeedSequence, BitGenerator, Generator}, optional
            A seed to initialize the `BitGenerator`. If `None`, then fresh,
            unpredictable entropy will be pulled from the OS. If an `int`
            is passed, then it will be passed to `SeedSequence` to derive
            the initial `BitGenerator` state. One may also pass in a `SeedSequence` instance.
            Additionally, when passed a `BitGenerator`, it will be wrapped by `Generator`.
            If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    tuple, made of
    - X_train, X_test: np.ndarray of predictors,
    - y_train, y_test: np.ndarray of responses,
    - Z_train: np.ndarray of environments,
    - f_train, f_test: np.ndarray of causal function.
    """

    rng = np.random.default_rng(seed)

    if q is None:
        q = min(p, r)

    M = generate_random_mat(p=p, r=r, q=q, eigengap=eigengap, seed=rng)
    W = np.eye(r)
    S = np.eye(p)
    # S = generate_random_covariance(
    #     ndims=p, max_condition_number=3, total_variance=p, seed=rng
    # )
    f_ = generate_tree(
        p_effective, tree_depth, min_leaf_width=0.75, scale_y=1.5, limit_x=2, seed=rng
    )
    gamma = generate_orthonormal_vector(ndims=p, seed=rng) * gamma_norm

    V_train = sample_Gaussian(n=n, ndims=p, S=S, seed=rng)
    Z_train = mean_Z + sample_Gaussian(n=n, ndims=r, S=W, seed=rng)
    noise_train = rng.normal(scale=sd_y, size=n)
    X_train = mean_X + Z_train @ M.T + V_train
    f_train = f_.predict(X_train[:, range(p_effective)])
    y_train = f_train + V_train @ gamma + noise_train

    # print(f"Sd of instruments: {np.var(Z_train @ M.T, axis=0)}")

    # print(f"Sd of f_train: {np.sqrt(np.var(f_train))}")
    # print(f"Sd of gamma*V: {np.sqrt(np.var(V_train @ gamma))}")
    # print(f"Sd of noise: {np.sqrt(np.var(noise_train))}")

    V_test = sample_Gaussian(n=n, ndims=p, S=S, seed=rng)
    Z_test = mean_Z + sample_Gaussian(n=n, ndims=r, S=W, seed=rng)
    noise_test = rng.normal(scale=sd_y, size=n)
    X_test = mean_X + interv_strength * Z_test @ M.T + V_test
    f_test = f_.predict(X_test[:, range(p_effective)])
    y_test = f_test + V_test @ gamma + noise_test

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        Z_train,
        Z_test,
        f_train,
        f_test,
        f_,
        M,
        S,
        gamma,
    )
