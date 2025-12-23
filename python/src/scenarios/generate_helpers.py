from typing import Optional, Tuple, Union

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence
from scipy.linalg import null_space, orth
from scipy.stats import ortho_group


def generate_random_projection(
    p: int,
    q: int,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> np.ndarray:
    """Create random projection matrix with `q` rows and `p` columns, with q <= p"""

    if p < q:
        raise Exception(
            "The projected dimension `q` cannot exceed the original dimension `p`."
        )

    rng = np.random.default_rng(seed)

    if p == 1:
        A = np.array([[1]])
    else:
        A = ortho_group.rvs(dim=p, random_state=rng)[:q, :]

    return A


def generate_orthonormal_vector(
    ndims: int,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> np.ndarray:
    """Uniformly sample from a `ndims-1`-sphere

    Parameters
    ----------
    ndims : int
        The number of dimensions

    seed : {None, int, SeedSequence, BitGenerator, Generator}, optional
            A seed to initialize the `BitGenerator`. If `None`, then fresh,
            unpredictable entropy will be pulled from the OS. If an `int`
            is passed, then it will be passed to `SeedSequence` to derive
            the initial `BitGenerator` state. One may also pass in a `SeedSequence` instance.
            Additionally, when passed a `BitGenerator`, it will be wrapped by `Generator`.
            If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    np.ndarray with `shape=(ndims,)`
        A random vector uniformly sampled from a `ndims`-sphere
    """

    rng = np.random.default_rng(seed)

    x = rng.normal(size=ndims)
    d = np.sqrt(np.sum(x**2))
    return x / d


def decompose_mat(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decomposes `mat` into its column space, `Q`, and its orthogonal complement, `R`

    Parameters
    ----------
    mat : np.ndarray of `shape=(p, q)`
        Matrix that we want to decompose. The number of rows `p` must be positive.

    Returns
    -------
    Q, R : Tuple(np.ndarray of `shape=(p, r)`, np.ndarray of `shape=(p, p-r)`)
        Matrix `Q` has columns corresponding to the directions spanned by columns of `mat`.
        Matrix `R` has columns in the orthogonal complement of `Q`.
    """
    p, q = mat.shape

    if p < 1:
        raise Exception("The number of rows of `mat` must be positive.")

    M = np.hstack([mat, np.zeros((p, 1))])
    return orth(M), null_space(M.T)


def problem_hardness(
    gamma: np.ndarray, W: np.ndarray, M: np.ndarray, S: np.ndarray
) -> float:
    """Computes hardness of the given SCM"""

    # TODO: does it makes sense or should remove it?

    A = M @ M.T
    B = np.linalg.inv(M @ W @ M.T + S) @ S
    d_ols = B @ gamma

    return np.sum((A @ d_ols) ** 2) / np.sum((d_ols) ** 2)


def distance_left_nullspace(M: np.ndarray, M_hat: np.ndarray) -> float:
    """Compute distance between the left null spaces of M and M_hat.

    Parameters
    ----------
    M : np.ndarray of `shape=(p, q)`
        A matrix.

    M_hat: np.ndarray of `shape=(p, q)`
        A matrix.

    Returns
    -------
    Distance between left null space of M and M_hat

    """

    # Compute left null space
    R = decompose_mat(M)[1]
    R_hat = decompose_mat(M_hat)[1]

    # Compute distance between null spaces
    return np.linalg.norm(R @ R.T - R_hat @ R_hat.T)  # type: ignore


def radial2D(
    num_basis: int,
    x_min=[-5, -5],
    x_max=[5, 5],
    sd_height=4.0,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
):
    # The function below was taken from https://github.com/sorawitj/HSIC-X/blob/master/experiments/distribution_generalization_nonlinear.py#L142
    # We modified some of the parameters.
    def radial2D_helper(X, centres, num_basis):
        Phi = np.zeros((X.shape[0], num_basis))
        for i in range(num_basis):
            Phi[:, i : i + 1] = np.exp(
                -1 * (np.linalg.norm(X - centres[i], axis=1, keepdims=True) / 3) ** 2
            )

        return Phi

    rng = np.random.default_rng(seed=seed)
    centres = rng.uniform(low=x_min, high=x_max, size=(num_basis, 2))
    w = rng.normal(0, sd_height, size=num_basis)
    f = lambda x: radial2D_helper(x, centres, num_basis) @ w

    return f
