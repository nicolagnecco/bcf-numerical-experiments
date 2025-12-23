# %%
from dataclasses import dataclass
from typing import Any, List

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from src.bcf.helpers import split_cont_cat, split_X_and_Z

ModelRegressor = Any
# %%


@dataclass
class AnchorRegression(BaseEstimator):
    """
    Anchor Regression Estimator.

    This estimator implements the [anchor regression methodology](https://arxiv.org/abs/1801.06229). It makes predictions under distributional shifts induced by exogenous/instrumental variables in the presence of unobserved confounding.

    Parameters:
    ----------
    n_exog : int
        Number of exogenous/instrumental variables.

    continuous_mask : NDArray[np.bool_]
        A boolean mask indicating which columns of X are continuous features.

    gamma : float >=0, default= 1.0
        Tuning parameter that penalizes the squared correlation between residuals and exogenous/instrumental variables.

    Attributes:
    ----------
    fy_xcat : LinearRegression or None
        Fitted regression model Y ~ X_categorical if there are categorical variables as determined by `continuous_mask`

    fy_xcont : LinearRegression
        Fitted regression model on transformed variables Y_tilde ~ X_tilde as described in Section 4.1 of https://arxiv.org/abs/1801.06229

    fy_z : LinearRegression
        Fitted regression model Y - fy_xcat(X_categorical) ~ Z

    fxcont_z : LinearRegression
        Fitted regression model X_cont ~ Z

    is_fitted_ : bool
        Indicator to check if the estimator has been fitted.

    Methods:
    --------
    fit(X, y):
        Fit the anchor regression model.

    predict(X):
        Predict using the fitted anchor regression model.

    """

    n_exog: int
    continuous_mask: NDArray[np.bool_]
    gamma: float = 1.0

    def __post_init__(self):
        """Initialization steps post object instantiation."""
        self.name = "anchor_regression"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed=None,
        check_input=True,
    ) -> None:
        """
        Fit the anchor regression model.

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
        # set up estimators
        self.fy_xcat = LinearRegression(fit_intercept=True)
        self.fy_z = LinearRegression(fit_intercept=True)
        self.fxcont_z = LinearRegression(fit_intercept=True)
        self.fy_xcont = LinearRegression(fit_intercept=True)

        # perform checks
        if check_input:
            X, y = check_X_y(X, y)

        # Split concatenated [X, Z] back into X and Z
        X_original, Z = split_X_and_Z(X, self.n_exog)

        # Store number of columns associated to X_original
        self.n_X_cols_ = X_original.shape[1]

        # Split continuous and categorical Xs
        X_cont, X_cat = split_cont_cat(self.continuous_mask, X_original)

        # step 1. regress y ~ X_cat and get residuals
        if X_cat.shape[1] > 0:
            self.fy_xcat.fit(X_cat, y)
            y_2 = y - self.fy_xcat.predict(X_cat)
        else:
            self.fy_xcat = None
            y_2 = y

        # step 2. regress y_2 ~ Z and X_cont ~ Z
        self.fy_z.fit(Z, y_2)
        self.fxcont_z.fit(Z, X_cont)

        # step 3. compute mean of y_ and X_cont
        self.y_2_mean_ = np.mean(y_2)
        self.X_cont_mean_ = np.mean(X_cont, axis=0)

        # step 4. compute transformed variables
        y_tilde = (
            self.y_2_mean_
            + (y_2 - self.fy_z.predict(Z))
            + self.gamma * (self.fy_z.predict(Z) - self.y_2_mean_)
        )
        X_tilde = (
            self.X_cont_mean_
            + (X_cont - self.fxcont_z.predict(Z))
            + self.gamma * (self.fxcont_z.predict(Z) - self.X_cont_mean_)
        )

        # step 5. regress y_tilde ~ X_tilde
        self.fy_xcont.fit(X_tilde, y_tilde)

        self.is_fitted_ = True

    def predict(self, X: np.ndarray, check_input=True) -> np.ndarray:
        """
        Predict using the anchor regression model.

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
            X_pred = split_X_and_Z(X, self.n_exog)[0]
        else:
            raise ValueError("Unexpected number of columns in the input data.")

        # Split continuous and categorical features from X_pred
        X_cont, X_cat = split_cont_cat(self.continuous_mask, X_pred)

        # predict target variable
        if self.fy_xcat is not None:
            y_cat = self.fy_xcat.predict(X_cat)
        else:
            y_cat = 0
        return y_cat + self.fy_xcont.predict(X_cont)


# %%
@dataclass
class AnchorRegressionCV(BaseEstimator):
    """
    Cross-validated Anchor Regression Estimator.

    This estimator cross-validates the gamma parameter of `AnchorRegression`.

    Parameters:
    ----------
    n_exog : int
        Number of exogenous/instrumental variables.

    continuous_mask : NDArray[np.bool_]
        A boolean mask indicating which columns of X are continuous features.

    gammas : NDArray[float], default=[0.0, 1.0, 10.0, 100.0]
        Tuning parameter that penalizes the squared correlation between residuals and exogenous/instrumental variables.

    Attributes:
    ----------
    f_yx : AnchorRegression
        Fitted anchor regression model with optimal gamma

    anchor_grid_search : GridSearchCV
        A fitted GridSearchCV object

    is_fitted_ : bool
        Indicator to check if the estimator has been fitted.

    Methods:
    --------
    fit(X, y):
        Fit the anchor regression model using 5-fold cross-validation over gammas.

    predict(X):
        Predict using the fitted anchor regression model.

    NOTE:
    ----
    This implementation uses standard K-fold cross validation via GridSearchCV and does not ensure that each level of the anchor appears only in one of the train/test folds.

    """

    n_exog: int
    continuous_mask: NDArray[np.bool_]
    gammas: List[float]

    def __post_init__(self):
        """Initialization steps post object instantiation."""
        self.name = "anchor_regression_cv"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed=None,
        check_input=True,
    ) -> None:

        # instatiate anchore regression object
        anchor_reg = AnchorRegression(
            n_exog=self.n_exog, continuous_mask=self.continuous_mask
        )

        # grid-search across gammas
        anchor_grid_search = GridSearchCV(
            anchor_reg, {"gamma": self.gammas}, scoring="neg_root_mean_squared_error"
        )

        anchor_grid_search.fit(X, y)

        # set the grid search object
        self.grid_search = anchor_grid_search

        # set best estimator
        self.f_yx = anchor_grid_search.best_estimator_

        self.is_fitted_ = True

    def predict(self, X: np.ndarray, check_input=True) -> np.ndarray:

        return self.f_yx.predict(X)


# %%
