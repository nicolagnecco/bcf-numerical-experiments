from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from src.bcf.reduced_rank_regression import RRR
from src.scenarios.generate_helpers import decompose_mat
from tqdm import tqdm
from xgboost import XGBRegressor

ModelRegressor = Any


@dataclass
class BCF(BaseEstimator):
    """
    Boosted Control Function (BCF) Estimator.

    This estimator implements the BCF method from the paper [Boosted Control Functions](https://arxiv.org/abs/2310.05805). It makes predictions under distributional shifts induced by exogenous/instrumental variables in the presence of unobserved confounding.

    Parameters:
    ----------
    n_exog : int
        Number of exogenous/instrumental variables.

    continuous_mask : NDArray[np.bool_]
        A boolean mask indicating which columns of X are continuous features.

    passes : int, default=2
        Number of boosting passes for the ensemble methods.

    fx : ModelRegressor, default=RandomForestRegressor()
        Regressor model for the function f(X).

    fx_imp : ModelRegressor, default=RandomForestRegressor()
        Regressor model for the invariant most predictive function based on X_continuous.

    gv : ModelRegressor, default=LinearRegression(fit_intercept=False)
        Regressor model for the function gamma(V).

    alphas : array-like, default=10 ** np.arange(-5.0, 6.0)
        List of alpha parameters for grid search to estimate the matrix M_0.

    verbose: bool, default=False
        Whether to print progress bar during fitting.

    Attributes:
    ----------
    M_0_ : ndarray
        The estimated matrix M_0.

    q_opt_ : int
        The estimated rank of M_0.

    R_ : ndarray
        Matrix spanning the null space of M_0 transpose.

    fx_ : ModelRegressor
        Fitted regressor model for the function f(X).

    fx_imp_ : ModelRegressor
        Fitted regressor model for the invariant most predictive function based on X_continuous.

    gv_ : ModelRegressor
        Fitted regressor model for the function gamma(V).

    is_fitted_ : bool
        Indicator to check if the estimator has been fitted.

    Methods:
    --------
    fit(X, y):
        Fit the BCF model.

    predict(X):
        Predict using the fitted BCF model.

    _split_X_and_Z(X, n_exog):
        Helper method to split the concatenated [X, Z] matrix into X and Z.

    _learn_M_0(X, Z, alphas):
        Helper method to learn the matrix `M_0`.

    _learn_ker_M_0_T(M_0):
        Helper method to learn the null space of `M_0.T`.


    """

    n_exog: int
    continuous_mask: NDArray[np.bool_]
    passes: int = 2
    fx: ModelRegressor = RandomForestRegressor()
    fx_imp: ModelRegressor = RandomForestRegressor()
    gv: ModelRegressor = LinearRegression(fit_intercept=False)
    alphas = 10 ** np.arange(-5.0, 6.0)
    verbose: bool = False

    def __post_init__(self):
        """Initialization steps post object instantiation."""
        self.name = "boosted_control_function"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed=None,
        check_input=True,
    ) -> None:
        """
        Fit the BCF model.

        Parameters:
        -----------
        X : numpy.ndarray
            The concatenated [X, Z] matrix, where `X` is the matrix of predictors
            and `Z` is the matrix of instruments.

            When the exogenous variables are categorical, the matrix `Z` must be encoded with `OneHotEncoder(drop="first", sparse=False)`.

        y : numpy.ndarray
            Target values.

        seed : int, optional
            Random seed for reproducibility.

        check_input : bool, default=True
            Whether to check input arrays.

        Returns:
        --------
        None
        """
        # clone estimators
        self.fx_ = clone(self.fx)
        self.fx_imp_ = clone(self.fx_imp)
        self.gv_ = clone(self.gv)

        # perform checks
        if check_input:
            X, y = check_X_y(X, y)

        # Split concatenated [X, Z] back into X and Z
        X_original, Z = self._split_X_and_Z(X, self.n_exog)

        # Store number of columns associated to X_original
        self.n_X_cols_ = X_original.shape[1]

        # Extract continuous features from X
        X_continuous = X_original[:, self.continuous_mask]

        # learn `M_0` and `q_opt_`, i.e., rank of M_0
        rrr = self._cross_validate_rrr(X_continuous, Z, self.alphas)
        self.M_0_, self.q_opt_, self.X_mean_ = rrr.M_hat_, rrr.k_opt_, rrr.X_mean_

        # compute control variables V
        V = X_continuous - rrr.predict(Z)

        # learn R (matrix spanning null space of M_0.T)
        self.R_ = self._learn_ker_M_0_T(self.M_0_)

        # Step 0
        f_X = np.mean(y)

        # Step 1
        for k in tqdm(range(self.passes), disable=not (self.verbose)):
            # Fit linear model using V
            y_ = y - f_X
            self.gv_.fit(V, y_)
            gamma_V = self.gv_.predict(V)

            # Fit flexible model using X
            y_ = y - gamma_V  # type: ignore
            self.fx_.fit(X_original, y_)
            f_X = self.fx_.predict(X_original)

        # Step 2
        y_ = y - f_X
        self.gv_.fit(V, y_)
        gamma_V = self.gv_.predict(V)

        # fit imp
        self.fx_imp_.fit((X_continuous - self.X_mean_) @ self.R_, y_)

        self.is_fitted_ = True

    def predict(self, X: np.ndarray, check_input=True) -> np.ndarray:
        """
        Predict using the BCF model.

        Parameters:
        -----------
        X : numpy.ndarray
            The predictors for which predictions are to be made.

        check_input : bool, default=True
            Whether to check the input array.

        Returns:
        --------
        numpy.ndarray
            Predicted values.
        """
        # check whether is fitted
        check_is_fitted(self, "is_fitted_")

        if check_input:
            # perform checks on X
            X = check_array(X, accept_sparse=True)

        # check whether the user passed [X, Z] or only X
        if X.shape[1] == self.n_X_cols_:
            X_pred = X
        elif X.shape[1] == self.n_X_cols_ + self.n_exog:
            X_pred = self._split_X_and_Z(X, self.n_exog)[0]
        else:
            raise ValueError("Unexpected number of columns in the input data.")

        # Extract continuous features from X
        X_continuous = X_pred[:, self.continuous_mask]

        # predict data
        return self.fx_.predict(X_pred) + self.fx_imp_.predict(
            (X_continuous - self.X_mean_) @ self.R_
        )

    @staticmethod
    def _split_X_and_Z(X: np.ndarray, n_exog: int) -> Tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def _cross_validate_rrr(X: np.ndarray, Z: np.ndarray, alphas: np.ndarray) -> RRR:
        """
        Return a fitted reduced rank regression object `RRR` with cross-validated parameter `alpha`.

        Parameters:
        -----------
        X (np.ndarray): The matrix of continuous features.

        Z (np.ndarray): The matrix of instrumental variables.

        alphas (np.ndarray): Regularization strength parameters for reduced rank regression.

        Returns:
        -----------
        RRR
            Fitted reduced rank regression object with cross-validated `alpha`.
        """
        parameters = {"alpha": alphas}

        red_rank_reg = RRR()

        clf = GridSearchCV(
            red_rank_reg, parameters, scoring="neg_root_mean_squared_error"
        )

        clf.fit(Z, X)

        return clf.best_estimator_

    @staticmethod
    def _learn_ker_M_0_T(M_0: np.ndarray) -> np.ndarray:
        """
        Learn the matrix spanning the null space of `M_0.T`.

        Parameters:
        -----------
        M_0 : numpy.ndarray
            The estimated matrix M_0 from the reduced rank regression.

        Returns:
        --------
        numpy.ndarray
            Matrix R spanning the null space of M_0 transpose. Returns a zero matrix if the null space is empty.
        """
        S, R = decompose_mat(M_0)

        if R.shape[1] == 0:
            return np.zeros((M_0.shape[0], 1))

        return R


@dataclass
class OLS(BaseEstimator):
    """Implementation of OLS learner control functions"""

    fx: ModelRegressor = RandomForestRegressor()

    def __post_init__(self):
        self.name = "ols"
        self.is_fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed=None,
        check_input=True,
    ) -> None:
        # clone estimators
        self.fx_ = clone(self.fx)

        if check_input:
            # perform checks
            X, y = check_X_y(X, y)

        # fit
        self.fx_.fit(X, y)

        self.is_fitted_ = True

    def predict(self, X: np.ndarray, check_input=True) -> np.ndarray:
        # check whether is fitted
        check_is_fitted(self, "is_fitted_")

        if check_input:
            # perform checks on X
            X = check_array(X, accept_sparse=True)

        return self.fx_.predict(X)
