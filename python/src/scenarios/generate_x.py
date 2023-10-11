from typing import Optional, Tuple, Union

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence
from scipy.stats import ortho_group
from sklearn.preprocessing import StandardScaler
from src.scenarios.generate_helpers import generate_orthonormal_vector

TOTAL_VARIANCE = 0.1
MAX_CONDITION_NUMBER = 1


def sample_X(
    n: int,
    p: int,
    r: int,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> np.ndarray:
    """Generate random realizations of the predictor vectors

    Parameters
    ----------
    n : int
        The number of observations.

    p : int
        The number of dimensions.

    r : int
        The rank of the covariance matrix between instruments/environments
        and predictors.

    seed : {None, int, SeedSequence, BitGenerator, Generator}, optional
            A seed to initialize the `BitGenerator`. If `None`, then fresh,
            unpredictable entropy will be pulled from the OS. If an `int`
            is passed, then it will be passed to `SeedSequence` to derive
            the initial `BitGenerator` state. One may also pass in a `SeedSequence` instance.
            Additionally, when passed a `BitGenerator`, it will be wrapped by `Generator`.
            If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    np.ndarray with `shape=(n, p)`
        The matrix of generated predictor values.
    """

    rng = np.random.default_rng(seed)

    mu = generate_mu(n=n, p=p, r=r, seed=rng)

    S = generate_random_covariance(
        ndims=p,
        max_condition_number=MAX_CONDITION_NUMBER,
        total_variance=TOTAL_VARIANCE,
        seed=rng,
    )

    V = sample_Gaussian(n=n, ndims=p, S=S, seed=rng)

    return mu + V


def sample_Z_sphere(
    n: int,
    r: int,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> np.ndarray:
    """TODO: comment function
    E[Z_j] = 0
    V[Z_j] = 1 / r
    COV[Z_j, Z_k] = 0
    r >= 1

    """

    rng = np.random.default_rng(seed)

    list_Z = list(
        map(lambda i: generate_orthonormal_vector(ndims=r, seed=rng), range(n))
    )

    return np.array(list_Z)


def sample_Z_onehot(
    n: int,
    r: int,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """TODO: comment function
    E[Z_j] = 0
    V[Z_j] = 1 / (r + 1)  - 1 / (r + 1)^2
    COV[Z_j, Z_k] = - 1 / (r + 1)^2
    r >= 1

    """

    rng = np.random.default_rng(seed)

    Z_sample_space = np.hstack([np.zeros((r, 1)), np.eye(r)])
    Z_cat = rng.choice(a=r + 1, size=n)
    Z = Z_sample_space[:, Z_cat].T

    Z_mean_centered = Z - np.repeat(a=1 / (r + 1), repeats=r)

    return Z_mean_centered, Z_cat


def sample_Gaussian(
    n: int,
    ndims: int,
    S: np.ndarray,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> np.ndarray:
    """Generate centred `ndims`-dimensional Gaussian noise with covariance `S`.

    Parameters
    ----------
    n : int
        The number of observations.

    ndims : int
        The number of dimensions.

    S : np.ndarray of size ndims x ndims
        A valid covariance matrix, i.e., positive definite.

    seed : {None, int, SeedSequence, BitGenerator, Generator}, optional
            A seed to initialize the `BitGenerator`. If `None`, then fresh,
            unpredictable entropy will be pulled from the OS. If an `int`
            is passed, then it will be passed to `SeedSequence` to derive
            the initial `BitGenerator` state. One may also pass in a `SeedSequence` instance.
            Additionally, when passed a `BitGenerator`, it will be wrapped by `Generator`.
            If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    np.ndarray with `shape=(n, ndims)`
        The matrix of generated noise terms.
    """
    rng = np.random.default_rng(seed)

    if ndims == 0:
        return np.empty(shape=(n, ndims))
    else:
        return rng.multivariate_normal(mean=np.zeros(ndims), cov=S, size=n)


def generate_mu(
    n: int,
    p: int,
    r: int,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> np.ndarray:
    """Generate the mean of the predictor vector

    Parameters
    ----------
    n : int
        The number of observations.

    p : int
        The number of dimensions.

    r : int
        The rank of the covariance matrix between instruments/environments
        and predictors.

    seed : {None, int, SeedSequence, BitGenerator, Generator}, optional
            A seed to initialize the `BitGenerator`. If `None`, then fresh,
            unpredictable entropy will be pulled from the OS. If an `int`
            is passed, then it will be passed to `SeedSequence` to derive
            the initial `BitGenerator` state. One may also pass in a `SeedSequence` instance.
            Additionally, when passed a `BitGenerator`, it will be wrapped by `Generator`.
            If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    np.ndarray with `shape=(n, p)`
        The matrix of generated means for the predictors
    """
    if r > p:
        raise Exception("Rank `r` cannot exceed the number of dimensions `p`.")

    rng = np.random.default_rng(seed)

    Z, Z_cat = sample_Z_onehot(n=n, r=r, seed=rng)
    M = generate_M(p, r, rng)
    mu = StandardScaler(with_std=False).fit_transform(X=Z) @ M

    return mu


def generate_random_mat(
    p: int,
    r: int,
    q: Optional[int] = None,
    eigengap: float = 1.0,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> np.ndarray:
    """Generate rank `q` random matrix

    Generates a matrix of the form

        `M = U @ D @ V.T`

    where `D` is a diagonal matrix with `q` entries with value `eigengap` and `min(p, r) - q` zeros.

    Parameters
    -------------
    p : int
        The number of rows.

    r : int
        The number of columns.

    q : int
        The rank of the matrix.

    eigengap: float
        The eigengap of the matrix.

    seed : {None, int, SeedSequence, BitGenerator, Generator}, optional
            A seed to initialize the `BitGenerator`. If `None`, then fresh,
            unpredictable entropy will be pulled from the OS. If an `int`
            is passed, then it will be passed to `SeedSequence` to derive
            the initial `BitGenerator` state. One may also pass in a `SeedSequence` instance.
            Additionally, when passed a `BitGenerator`, it will be wrapped by `Generator`.
            If passed a `Generator`, it will be returned unaltered.

    Returns
    -------------
    np.ndarray with `shape=(p, r)`
         The generated matrix
    """

    if q is None:
        q = min(p, r)

    if q > min(p, r):
        raise Exception(
            "Rank `q` cannot exceed the smallest of rows `p` and columns `r`."
        )

    rng = np.random.default_rng(seed)

    U = safe_ortho(dim=p, random_state=rng)
    V = safe_ortho(dim=r, random_state=rng)
    diag_vals = np.concatenate(
        [np.repeat(a=eigengap, repeats=q), np.repeat(a=0, repeats=min(p, r) - q)]
    )
    D = np.zeros(shape=(p, r))
    np.fill_diagonal(a=D, val=diag_vals)

    return U @ D @ V.T


def generate_M(
    p: int,
    r: int,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> np.ndarray:
    """TODO: comment function"""

    # TODO: handle case where r > p.

    if r > p:
        raise Exception("Rank `r` cannot exceed the number of dimensions `p`.")

    rng = np.random.default_rng(seed)
    if p == 1:
        return np.ones(shape=(p, r))
    else:
        M = ortho_group.rvs(dim=p, random_state=rng)
        return M[:, :r]


def generate_random_covariance(
    ndims: int,
    max_condition_number: int,
    total_variance: float,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> np.ndarray:
    """Generate random covariance matrix

    First, samples a diagonal matrix D with shape `(ndims, ndims)`.
    The eigenvalues on the diagonal are sampled uniformly, i.e.,
    `lambda_j ~ U([1, max_condition_number])` and rescaled
    as `lambda_j * 2 * total_variance / ((1 + max_condition_number) * ndims)`,
    so that, on average, `trace(D) = total_variance`.
    Second, samples an orthogonal matrix Q with shape `(ndims, ndims)` from the O(N)
    Haar distribution (the only uniform distribution on O(N)).
    The resulting covariance matrix is Q*D*Q^T.

    Parameters
    ----------
    ndims : int
        The number of dimensions.

    max_condition_number: int
        Max condition number of the resulting covariance matrix.

    total_variance: float
        Sum of variances of the resulting covariance matrix.

    seed : {None, int, SeedSequence, BitGenerator, Generator}, optional
            A seed to initialize the `BitGenerator`. If `None`, then fresh,
            unpredictable entropy will be pulled from the OS. If an `int`
            is passed, then it will be passed to `SeedSequence` to derive
            the initial `BitGenerator` state. One may also pass in a `SeedSequence` instance.
            Additionally, when passed a `BitGenerator`, it will be wrapped by `Generator`.
            If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    np.ndarray with `shape=(ndims, ndims)`.
        A random covariance matrix
    """

    rng = np.random.default_rng(seed)

    mean_unif = (1 + max_condition_number) / 2
    eigvals = (
        rng.uniform(low=1, high=max_condition_number, size=ndims)
        / (mean_unif * ndims)
        * total_variance
    )

    D = np.diag(v=eigvals)

    if ndims == 0:
        Q = np.empty(shape=(0, 0))
    elif ndims == 1:
        Q = np.array([[1]])
    else:
        Q = ortho_group.rvs(dim=ndims, random_state=rng)

    return Q @ D @ Q.T


def safe_ortho(
    dim: int,
    random_state: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> np.ndarray:
    if dim == 0:
        Q = np.empty(shape=(0, 0))
    elif dim == 1:
        Q = np.array([[1]])
    else:
        Q = ortho_group.rvs(dim=dim, random_state=random_state)

    return Q
