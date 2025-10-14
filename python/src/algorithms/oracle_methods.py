from dataclasses import dataclass
from typing import Any, Callable, Literal, Union

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from src.bcf.helpers import split_X_and_Z
from src.scenarios.generate_helpers import decompose_mat
from src.utils.data_definitions import Tree
from xgboost import XGBRegressor

# TODO: attributes estimated from data (i.e., set by fit()): append "_"; private methods/properties: prepend "_"


@dataclass
class CausalFunction:
    """Oracle for causal function"""

    causal_function: Union[Tree, Any] = Tree()
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

    causal_function: Union[Tree, Any] = Tree()
    instrument_matrix: np.ndarray = np.array([])
    confounder_cov: np.ndarray = np.array([])
    confounder_effect: np.ndarray = np.array([])
    name: str = "imp_function"
    is_fitted_: bool = False

    def fit(self, X, y, Z, seed=None):
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
class IMPFunctionNonLin:
    """Oracle for IMP function when control function is possibly nonlinear"""

    causal_function: Callable[[np.ndarray], np.ndarray]
    n_exog: int
    confounder_effect: Callable[[np.ndarray], np.ndarray]
    instrument_matrix: np.ndarray = np.array([])
    confounder_cov: np.ndarray = np.array([])
    mode: Literal["exact", "computed"] = "exact"
    boosted_estimator: Union[RandomForestRegressor, XGBRegressor] = XGBRegressor()
    use_imp: bool = True
    name: str = "imp_function"
    is_fitted_: bool = False

    def fit(self, X, y, seed=None):
        # perform checks
        X, y = check_X_y(X, y)
        n, p = X.shape

        # split X and Z
        X, Z = split_X_and_Z(X, self.n_exog)

        # center X
        self.X_mean = np.mean(X, axis=0)
        X_centered = X - self.X_mean

        # set theoretical quantities
        M = self.instrument_matrix
        S = self.confounder_cov

        # compute imp
        if M.shape[1] > 0 and M.shape[1] <= p:
            Q, R = decompose_mat(M)
            self.R_ = R
            self._use_imp_ = True
        else:
            self._use_imp_ = False

        self._use_imp_ = self._use_imp_ and self.use_imp

        # Compute V
        V = X - Z @ M.T

        if self._use_imp_:
            if self.mode == "exact":
                self.delta_ = R @ np.linalg.inv(R.T @ S @ R) @ R.T @ S
                # print(f"Delta IMP: {self.delta_}")
            else:
                self.boosted_estimator_ = clone(self.boosted_estimator)
                self.boosted_estimator_.fit(X @ R, self.confounder_effect(V))

        self.is_fitted_ = True

    def predict(self, X):
        # perform checks
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")  # type: ignore

        # predict data
        if self._use_imp_:
            if self.mode == "exact":
                return (
                    self.causal_function(X)
                    + self.confounder_effect((X @ self.delta_)).ravel()
                )
            else:
                return self.causal_function(X) + self.boosted_estimator_.predict(
                    X @ self.R_
                )
        else:
            return self.causal_function(X)


@dataclass
class ConstantFunc:
    """Constant Function"""

    name: str = "constant_func"
    is_fitted_: bool = False

    def fit(self, X, y, Z, seed=None):
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
