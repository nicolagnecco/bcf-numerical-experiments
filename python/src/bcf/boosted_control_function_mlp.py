import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from pytorch_lightning import seed_everything
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from src.bcf.helpers import split_cont_cat, split_X_and_Z
from src.bcf.mlp import (MLP, predict_full, predict_fx_only, train_fx,
                         train_fx_gv, train_fx_imp)
from src.bcf.reduced_rank_regression import cross_validate_rrr, learn_ker_M_0_T


@dataclass
class BCFMLP(BaseEstimator):
    n_exog: int
    continuous_mask: NDArray[np.bool_]
    fx_factory: Callable[[int], MLP]  # lambda x: MLP(in_dim=x)
    fx_imp_factory: Callable[[int], MLP]  # lambda x: MLP(in_dim=x)
    gv_factory: Callable[[int], MLP]  # lambda x: MLP(in_dim=x)
    alphas: np.ndarray = field(default_factory= lambda: 10 ** np.arange(-5.0, 6.0))
    predict_imp: bool = True
    weight_decay_f: float = 1e-4
    weight_decay_g: float = 1e-4
    weight_decay_fimp: float = 0e-4
    lr_f: float = 1e-3
    lr_g: float = 1e-3
    lr_fimp: float = 1e-4
    epochs_step_1: int = 100
    epochs_step_2: int = 100

    def __post_init__(self):
        """Initialization steps post object instantiation."""
        self.name = "boosted_control_function_mlp"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed=None,
        check_input=True,
    ) -> None:

        # 1) Validate & split
        X, Z, y, X_cont, X_cat = validate_and_split(
            X, y, self.n_exog, self.continuous_mask, check_input
        )
        self.n_X_cols_ = X.shape[1]

        # 2) Learn RRR structures
        rrr, self.M_0_, self.q_opt_, self.R_, V = learn_rrr_and_structures(
            X_cont, Z, self.alphas
        )
        self.n_R_cols_ = self.R_.shape[1]
        self.n_V_cols_ = V.shape[1]

        # 3) Use IMP if R has at least one non-zero entry
        self._use_imp_ = bool(self.predict_imp and np.any(self.R_))

        # 4) Scale features/targets (storing means and std)
        self.X_mean_cont_ = np.mean(X_cont, axis=0)
        self.X_std_cont_ = np.std(X_cont, axis=0)
        X_cont_scaled = scale_matrix(X_cont, self.X_mean_cont_, self.X_std_cont_)
        X_scaled = np.concatenate([X_cont_scaled, X_cat], axis=1)
        X_cont_rot = rotate(X_cont_scaled, self.R_)  # input to fx_imp
        self.y_mean_ = float(y.mean())
        y_centered = center_y(y, self.y_mean_)
        self.y_std_ = float(np.std(y_centered)) or 1.0
        y_scaled = y_centered / self.y_std_

        # 5) Learn BCF
        # 5.0) Set seeds
        if seed is not None:
            seed_everything(
                seed, workers=False
            )  # NOTE: to ensure reproducibility when num_workers > 0 in DataLoaders, see https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader

        # 5.1) instanstiate estimators
        self.fx_ = self.fx_factory(self.n_X_cols_)
        self.fx_imp_ = self.fx_imp_factory(self.n_R_cols_)
        self.gv_ = self.gv_factory(self.n_V_cols_)

        # 5.2) Learn f(X) + g(V)
        train_fx_gv(
            fx=self.fx_,
            gv=self.gv_,
            X=torch.from_numpy(X_scaled).float(),
            V=torch.from_numpy(V).float(),
            y=torch.from_numpy(y_scaled).float(),
            weight_decay_f=self.weight_decay_f,
            weight_decay_g=self.weight_decay_g,
            lr_f=self.lr_f,
            lr_g=self.lr_g,
            epochs=self.epochs_step_1,
        )

        # 5.3) Learn f_imp(RX)
        if self._use_imp_:
            train_fx_imp(
                fx=self.fx_,
                fx_imp=self.fx_imp_,
                X=torch.from_numpy(X_scaled).float(),
                RX=torch.from_numpy(X_cont_rot).float(),
                y=torch.from_numpy(y_scaled).float(),
                weight_decay=self.weight_decay_fimp,
                lr=self.lr_fimp,
                epochs=self.epochs_step_2,
            )

        self.is_fitted_ = True

    def predict(self, X: np.ndarray, check_input=True) -> np.ndarray:
        # check whether is fitted
        check_is_fitted(self, "is_fitted_")
        if check_input:
            X = check_array(X, accept_sparse=True)

        # accept X or [X,Z]
        if X.shape[1] == self.n_X_cols_:
            X_pred = X
        elif X.shape[1] == self.n_X_cols_ + self.n_exog:
            X_pred, _ = split_X_and_Z(X, self.n_exog)
        else:
            raise ValueError(
                f"Unexpected n_cols={X.shape[1]} (expected {self.n_X_cols_} or {self.n_X_cols_ + self.n_exog})"
            )

        # Split continuous and categorical features from X
        X_cont, X_cat = split_cont_cat(self.continuous_mask, X_pred)

        # standardize Xs
        X_cont_scaled = scale_matrix(X_cont, self.X_mean_cont_, self.X_std_cont_)
        X_scaled = np.concatenate([X_cont_scaled, X_cat], axis=1)  # input to self.fx_
        X_cont_rot = rotate(X_cont_scaled, self.R_)  # input to self.fx_imp_

        if self._use_imp_:
            y_hat_scaled = predict_full(
                fx=self.fx_,
                fx_imp=self.fx_imp_,
                X=torch.from_numpy(X_scaled).float(),
                RX=torch.from_numpy(X_cont_rot).float(),
                y_mean=0.0,
            )
            y_pred = self.y_mean_ + self.y_std_ * y_hat_scaled
        else:
            y_hat_scaled = predict_fx_only(
                fx=self.fx_, X=torch.from_numpy(X_scaled).float(), y_mean=0.0
            )

            y_pred = self.y_mean_ + self.y_std_ * y_hat_scaled

        # predict data
        return y_pred


@dataclass
class OLSMLP(BaseEstimator):
    """Implementation of OLS"""

    fx_factory: Callable[[int], MLP]
    continuous_mask: NDArray[np.bool_]
    weight_decay: float = 1e-4
    lr: float = 1e-3
    epochs: int = 100

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

        # compute number predictors
        self.n_X_cols_ = X.shape[1]

        # split into continuous and categorical
        X_cont, X_cat = split_cont_cat(self.continuous_mask, X)

        # scale X and y
        self.X_mean_cont_ = np.mean(X_cont, axis=0)
        self.X_std_cont_ = np.std(X_cont, axis=0, ddof=0)
        X_cont_scaled = scale_matrix(X_cont, self.X_mean_cont_, self.X_std_cont_)
        X_scaled = np.concatenate([X_cont_scaled, X_cat], axis=1)
        self.y_mean_ = float(y.mean())
        y_centered = center_y(y, self.y_mean_)
        self.y_std_ = float(np.std(y_centered)) or 1.0
        y_scaled = y_centered / self.y_std_

        # set seeds
        if seed is not None:
            seed_everything(
                seed, workers=False
            )  # NOTE: to ensure reproducibility when num_workers > 0 in DataLoaders, see https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader

        # instantiate neural net
        self.fx_ = self.fx_factory(self.n_X_cols_)

        # train fx_
        train_fx(
            fx=self.fx_,
            X=torch.from_numpy(X_scaled).float(),
            y=torch.from_numpy(y_scaled).float(),
            weight_decay=self.weight_decay,
            lr=self.lr,
            epochs=self.epochs,
        )

        self.is_fitted_ = True

    def predict(self, X: np.ndarray, check_input=True) -> np.ndarray:
        # check whether is fitted
        check_is_fitted(self, "is_fitted_")
        if check_input:
            X = check_array(X, accept_sparse=True)

        # Split continuous and categorical features from X
        X_cont, X_cat = split_cont_cat(self.continuous_mask, X)

        # standardize Xs
        X_cont_scaled = scale_matrix(X_cont, self.X_mean_cont_, self.X_std_cont_)
        X_scaled = np.concatenate([X_cont_scaled, X_cat], axis=1)  # input to self.fx_

        y_hat_scaled = predict_fx_only(
            fx=self.fx_,
            X=torch.from_numpy(X_scaled).float(),
            y_mean=0.0,
        )

        y_pred = self.y_mean_ + self.y_std_ * y_hat_scaled

        # predict data
        return y_pred


# ---- Helpers
def validate_and_split(X, y, n_exog, continuous_mask, check_input: bool):
    if check_input:
        X, y = check_X_y(X, y)
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    X_original, Z = split_X_and_Z(X, n_exog)

    X_cont, X_cat = split_cont_cat(continuous_mask, X_original)

    return X_original, Z, y, X_cont, X_cat


def learn_rrr_and_structures(X_cont: np.ndarray, Z: np.ndarray, alphas: np.ndarray):
    """Fit RRR, get M_0, q_opt, nullspace R, controls V."""
    rrr = cross_validate_rrr(X_cont, Z, alphas)
    M0, q = rrr.M_hat_, rrr.k_opt_
    R = learn_ker_M_0_T(M0)
    V = X_cont - rrr.predict(Z)
    return rrr, M0, q, R, V


def center_y(y: np.ndarray, mu: float) -> np.ndarray:
    return (y - mu).reshape(-1, 1)


def rotate(X_cont_scaled: np.ndarray, R: np.ndarray):
    return X_cont_scaled @ R if R.size else np.zeros((X_cont_scaled.shape[0], 0))


def scale_matrix(X: np.ndarray, X_mean: np.ndarray, X_std: np.ndarray) -> np.ndarray:
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std

    return X_scaled
