from dataclasses import dataclass

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from src.scenarios.generate_helpers import decompose_mat
from src.utils.data_definitions import Tree

# TODO: attributes estimated from data (i.e., set by fit()): append "_"; private methods/properties: prepend "_"


@dataclass
class CausalFunction:
    """Oracle for causal function"""

    causal_function: Tree = Tree()
    name: str = "causal_function"
    is_fitted_: bool = False

    def fit(self, X, y, Z):
        # perform checks
        X, y = check_X_y(X, y)
        Z, y = check_X_y(Z, y)
        n, p = X.shape

        self.is_fitted_ = True

    def predict(self, X):
        # perform checks
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")  # type: ignore

        # predict data
        return self.causal_function.predict(X)


@dataclass
class IMPFunction:
    """Oracle for IMP function"""

    causal_function: Tree = Tree()
    instrument_matrix: np.ndarray = np.array([])
    confounder_cov: np.ndarray = np.array([])
    confounder_effect: np.ndarray = np.array([])
    name: str = "imp_function"
    is_fitted_: bool = False

    def fit(self, X, y, Z):
        # perform checks
        X, y = check_X_y(X, y)
        Z, y = check_X_y(Z, y)
        n, p = X.shape

        M = self.instrument_matrix
        S = self.confounder_cov
        gamma = self.confounder_effect

        # compute imp
        if M.shape[1] > 0 and M.shape[1] <= p:
            Q, R = decompose_mat(M)
            self.delta_ = R @ np.linalg.inv(R.T @ S @ R) @ R.T @ S @ gamma
            # print(f"Delta IMP: {self.delta_}")
        else:
            self.delta_ = np.zeros(shape=(p,))

        self.is_fitted_ = True

    def predict(self, X):
        # perform checks
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")  # type: ignore

        # predict data
        return self.causal_function.predict(X) + X @ self.delta_


@dataclass
class ConstantFunc:
    """Constant Function"""

    name: str = "constant_func"
    is_fitted_: bool = False

    def fit(self, X, y, Z):
        # perform checks
        X, y = check_X_y(X, y)
        Z, y = check_X_y(Z, y)
        n, p = X.shape

        self.c = np.mean(y)

        self.is_fitted_ = True

    def predict(self, X):
        # perform checks
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")  # type: ignore

        # predict data
        n, p = X.shape

        return np.repeat(self.c, n)
