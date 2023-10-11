from typing import Optional, Tuple, Union

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence
from src.scenarios.generate_helpers import decompose_mat, generate_orthonormal_vector
from src.utils.data_definitions import Tree


def generate_gamma(
    M: np.ndarray,
    r: int,
    S: np.ndarray,
    W: np.ndarray,
    gamma_norm: float,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> np.ndarray:
    """Generate `gamma` vector

    The `gamma` vector controls how much `V` coufounds the predictors `X`
    and the response `y`.

    ASSUME: Z ~ N(0, W) indep V ~ N(0, S)

    Parameters
    ----------
    M : np.ndarray of `shape=(p, r)`
        Orthonormal matrix where the columns are the directions spanned by
        the environments.

    r : int
        The rank of the covariance matrix between instruments/environments
        and predictors.

    S : np.ndarray of `shape=(p, p)`
        The covariance matrix of `V`.

    W: np.ndarray of `shape=(r, r)`
        The covariance matrix of `Z`.

    gamma_norm: float > 0
        The strength of the confounder. It is expressed as the norm of the eigenvectors
        of `M M^T (M W M^T + S)^{-1} S`.

    seed : {None, int, SeedSequence, BitGenerator, Generator}, optional
            A seed to initialize the `BitGenerator`. If `None`, then fresh,
            unpredictable entropy will be pulled from the OS. If an `int`
            is passed, then it will be passed to `SeedSequence` to derive
            the initial `BitGenerator` state. One may also pass in a `SeedSequence` instance.
            Additionally, when passed a `BitGenerator`, it will be wrapped by `Generator`.
            If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    `gamma`: np.ndarray of `shape=(p,)`
        The resulting `gamma` vector.
    """

    # TODO: to understand (if does not make sense remove it)

    rng = np.random.default_rng(seed)
    p = S.shape[0]

    # if r = 0, the problem difficulty is 0 for any gamma in R^p.
    if r == 0:
        return np.zeros(p)

    # Define matrix
    chi = M @ M.T @ np.linalg.inv(M @ W @ M.T + S) @ S
    U, D, V = np.linalg.svd(chi)
    a, b = np.linalg.eig(chi)

    # Pick randomly eigenvector/eigenvalue between 1 and r
    rand_pos = rng.choice(a=r)
    random_eig = U[:, rand_pos] / D[rand_pos]

    # Return rescaled eigenvector
    return random_eig * gamma_norm


def sample_value(
    scale: float,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> float:
    """TODO: comment this function"""

    rng = np.random.default_rng(seed)

    return rng.normal(scale=scale, size=1).item()


def sample_feature(
    features: np.ndarray,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> Tuple[int, int]:
    """Sample one feature from features and return its position"""

    rng = np.random.default_rng(seed)

    feature = rng.choice(a=features, size=1).item()
    feature_pos = np.where(features == feature)

    return feature, feature_pos


def sample_threshold(
    low: float,
    high: float,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> float:
    """TODO: comment this function"""

    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=1).item()


def generate_tree(
    ndims: int,
    depth: int,
    min_leaf_width: float,
    scale_y: float,
    limit_x: float,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> Tree:
    """TODO: comment this function"""

    if ndims < 1:
        raise Exception("Number of dimensions `ndims` must be positive.")

    if depth < 0:
        raise Exception("Tree `depth` cannot be negative.")

    rng = np.random.default_rng(seed)
    features = np.arange(ndims)
    lows = -np.ones(ndims) * limit_x
    highs = np.ones(ndims) * limit_x

    # TODO: in a wrapper function called only *once*, from here ...

    # check lows.shape = (ndims, )
    # check highs.shape = (ndims, )
    # check lows[0] < highs[0]

    # until here

    return generate_tree_helper(
        depth, features, lows, highs, min_leaf_width, scale_y, rng
    )


def generate_tree_helper(
    depth: int,
    features: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    min_leaf_width: float,
    scale_y: float,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> Tree:
    """TODO: comment this function"""

    rng = np.random.default_rng(seed)

    if depth == 0 or features.size == 0:
        # you reached a leaf, so sample a value
        return Tree(value=sample_value(scale=scale_y, seed=rng))
    else:
        # sample a feature j
        j, j_pos = sample_feature(features=features, seed=rng)

        if highs[j] - lows[j] < min_leaf_width:
            # if leaf width is too small in feature j, remove feature and call function again
            features = np.delete(arr=features, obj=j_pos)
            return generate_tree_helper(
                depth=depth,
                features=features,
                lows=lows,
                highs=highs,
                min_leaf_width=min_leaf_width,
                scale_y=scale_y,
                seed=rng,
            )
        else:
            # sample a threshold value for feature j
            t = sample_threshold(
                low=lows[j] + min_leaf_width / 2,
                high=highs[j] - min_leaf_width / 2,
                seed=rng,
            )

            # create limits for left child
            lows_left = np.copy(lows)
            highs_left = np.copy(highs)
            highs_left[j] = t

            # create limits for right child
            lows_right = np.copy(lows)
            highs_right = np.copy(highs)
            lows_right[j] = t

            # recurse on the two children
            return Tree(
                feature=j,
                threshold=t,
                left=generate_tree_helper(
                    depth=depth - 1,
                    features=features,
                    lows=lows_left,
                    highs=highs_left,
                    min_leaf_width=min_leaf_width,
                    scale_y=scale_y,
                    seed=rng,
                ),
                right=generate_tree_helper(
                    depth=depth - 1,
                    features=features,
                    lows=lows_right,
                    highs=highs_right,
                    min_leaf_width=min_leaf_width,
                    scale_y=scale_y,
                    seed=rng,
                ),
            )
