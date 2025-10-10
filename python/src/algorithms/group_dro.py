# %%
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import LinearModel
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from src.algorithms.group_dro_utils import (
    MyLossComputer,
    fit_groupdro,
    make_loader_from_numpy,
)
from src.bcf.helpers import split_cont_cat, split_X_and_Z
from src.bcf.mlp import MLP
from src.bcf.reduced_rank_regression import cross_validate_rrr, learn_ker_M_0_T
from tqdm import tqdm

ModelRegressor = Any
# %%


@dataclass
class GroupDRO(BaseEstimator):

    fx_factory: Callable[[int], MLP]  # lambda x: MLP(in_dim=x)
    n_groups: int
    lr: float = 1e-3
    step_size: float = 0.01
    batch_size: int = 128
    n_epochs: int = 20
    val_fraction: float = 0.2
    weight_decay: float = 0.0
    device: Optional[torch.device] = None
    verbose: bool = True
    seed: Optional[int] = None

    def __post_init__(self):
        """Initialization steps post object instantiation."""
        self.name = "group_dro"
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _split_train_val(self, X, y, g):
        N = X.shape[0]
        idx = np.arange(N)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(idx)
        n_val = int(self.val_fraction * N)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        return (X[tr_idx], y[tr_idx], g[tr_idx]), (X[val_idx], y[val_idx], g[val_idx])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        g: np.ndarray,
        seed=None,
        check_input=True,
    ) -> None:

        # create MLP
        self.n_X_cols_ = X.shape[1]
        self.fx_ = self.fx_factory(self.n_X_cols_)

        # set seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # set up loaders
        (X_tr, y_tr, g_tr), (X_val, y_val, g_val) = self._split_train_val(X, y, g)
        train_loader, _ = make_loader_from_numpy(
            X_tr,
            y_tr,
            g_tr,
            batch_size=self.batch_size,
            shuffle=True,
            device=self.device,
        )
        val_loader, _ = make_loader_from_numpy(
            X_val,
            y_val,
            g_val,
            batch_size=self.batch_size,
            shuffle=False,
            device=self.device,
        )

        # train model
        fit_groupdro(
            self.fx_,
            train_loader,
            val_loader,
            self.n_groups,
            self.n_epochs,
            lr=self.lr,
            step_size=self.step_size,
            wd=self.weight_decay,
            device=self.device,
        )

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

        self.fx_.eval()

        model_device = next(self.fx_.parameters()).device
        X_t = torch.as_tensor(X, dtype=torch.float32, device=model_device)
        with torch.no_grad():
            yhat = self.fx_(X_t)
            if yhat.dim() == 2 and yhat.size(-1) == 1:
                yhat = yhat.squeeze(-1)

        return yhat.cpu().numpy()
