# %%
import argparse
import logging
import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from src.bcf.boosted_control_function_2 import BCF, OLS
from xgboost import XGBRegressor

# Constants and function definitions
DATA_PATH = "../data/processed/"
RESULT_PATH = "../../results/output_data/"
SAVE_PREDICTIONS = True
ONLY_X_AT_PREDICTION = True  # Set to True to use only X, and False to use [X, Z]
USE_WHOLE_DATA = True  # Set to True if don't split train/test
MASK = np.repeat(True, 2)


def load_data(path: str):
    dat = pd.read_csv(filepath_or_buffer=path)

    y = dat["Y"]
    Z = dat[["E"]]
    X = dat.drop(["Y", "E"], axis=1)

    return X, y, Z


def eval_model(model: Union[BCF, OLS], X_test: NDArray, y_test: NDArray):
    y_hat = model.predict(X_test)
    mse = ((y_hat - y_test) ** 2).mean()
    return mse


def mse_standard_error(
    model: Union[BCF, OLS], X_test: NDArray, y_test: NDArray
) -> float:
    """Compute the standard error of the mean squared error."""

    y_hat = model.predict(X_test)
    squared_errors = (y_hat - y_test) ** 2
    se = squared_errors.std() / np.sqrt(len(y_hat))
    return se


def get_model(model_type: List[str], base_estimator: BaseEstimator, param_grid=None):
    if model_type == "grid_search" and param_grid:
        return GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=5,
        )
    return base_estimator


def define_fit_models(
    X_train: pd.DataFrame, y_train: pd.Series, Z_train: pd.DataFrame
) -> Tuple[BCF, BCF, OLS, OLS]:
    # BCF
    bcf = BCF(
        n_exog=Z_train.shape[1],
        continuous_mask=MASK,
        fx=Ridge(),
        gv=Ridge(),
        fx_imp=RandomForestRegressor(),
        passes=2,
    )

    bcf.fit(np.hstack([X_train, Z_train]), y_train.ravel())

    print(f"Rank M_0: {bcf.q_opt_}")

    # Anchor
    bcf_lin = BCF(
        n_exog=Z_train.shape[1],
        continuous_mask=MASK,
        fx=Ridge(),
        gv=Ridge(),
        fx_imp=Ridge(),
        passes=2,
    )

    bcf_lin.fit(np.hstack([X_train, Z_train]), y_train.ravel())

    # Input covariates for other models
    input_covariates = (
        X_train.to_numpy() if ONLY_X_AT_PREDICTION else np.hstack([X_train, Z_train])
    )

    # LS
    ls = OLS(fx=RandomForestRegressor())

    ls.fit(input_covariates, y_train.ravel())

    # OLS
    ols = OLS(fx=Ridge())
    ols.fit(input_covariates, y_train.ravel())

    # Ave

    return bcf, bcf_lin, ls, ols


def onehotencode(
    Z_train: pd.DataFrame, Z_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Initialize OneHotEncoder
    encoder = OneHotEncoder(drop="first", sparse=False)

    # Fit and transform data
    Z_train_ = encoder.fit_transform(Z_train)
    Z_test_ = encoder.transform(Z_test)

    # Number of columns in the transformed data
    num_envs = Z_train_.shape[1]

    # Create column names
    column_names = [f"Z_{i + 1}" for i in range(num_envs)]

    # Convert to DataFrame
    return (
        pd.DataFrame(Z_train_, columns=column_names),
        pd.DataFrame(Z_test_, columns=column_names),
    )


def predict_models(
    bcf: BCF, bcf_lin: BCF, ls: OLS, ols: OLS, X: pd.DataFrame, Z: pd.DataFrame
) -> pd.DataFrame:
    # BCF
    f_0 = bcf.fx_.predict(X)

    columns_to_select = X.columns[bcf.continuous_mask].tolist()
    filtered_df = X[columns_to_select]

    f_imp = bcf.fx_imp_.predict((filtered_df - bcf.X_mean_) @ bcf.R_)

    f_bcf = bcf.predict(X.to_numpy())

    # BCF-lin
    f_bcf_lin = bcf_lin.predict(X.to_numpy())

    # Input covariates for other models
    input_covariates = X.to_numpy() if ONLY_X_AT_PREDICTION else np.hstack([X, Z])

    # LS
    f_ls = ls.predict(input_covariates)

    # OLS
    f_ols = ols.predict(input_covariates)

    return pd.DataFrame(
        {
            "f_0": f_0,
            "IMP": f_imp,
            "BCF": f_bcf,
            "BCF-lin": f_bcf_lin,
            "LS": f_ls,
            "OLS": f_ols,
        },
        index=X.index,
    )


def eval_models(bcf, bcf_lin, ls, ols, X, y, Z, y_train, rep) -> List:
    # Input covariates for models other than BCF
    input_covariates = X if ONLY_X_AT_PREDICTION else np.hstack([X, Z])

    bcf_mse = eval_model(bcf, np.hstack([X, Z]), y)
    bcf_lin_mse = eval_model(bcf_lin, np.hstack([X, Z]), y)
    ls_mse = eval_model(ls, input_covariates, y)
    ols_mse = eval_model(ols, input_covariates, y)
    ave_mse = ((y - np.mean(y_train)) ** 2).mean()

    ave_standard_error = ((y - np.mean(y_train)) ** 2).std() / np.sqrt(len(y))

    print(f"BCF: {bcf_mse}")
    print(f"BCF-lin: {bcf_lin_mse}")
    print(f"LS: {ls_mse}")
    print(f"OLS: {ols_mse}")
    print(f"Ave: {ave_mse}")

    return [
        {
            "Rep": rep,
            "Method": "BCF",
            "MSE": bcf_mse,
            "MSE_standard_error": mse_standard_error(bcf, np.hstack([X, Z]), y),
        },
        {
            "Rep": rep,
            "Method": "BCF-lin",
            "MSE": bcf_lin_mse,
            "MSE_standard_error": mse_standard_error(bcf_lin, np.hstack([X, Z]), y),
        },
        {
            "Rep": rep,
            "Method": "LS",
            "MSE": ls_mse,
            "MSE_standard_error": mse_standard_error(ls, input_covariates, y),
        },
        {
            "Rep": rep,
            "Method": "OLS",
            "MSE": ols_mse,
            "MSE_standard_error": mse_standard_error(ols, input_covariates, y),
        },
        {
            "Rep": rep,
            "Method": "AVE",
            "MSE": ave_mse,
            "MSE_standard_error": ave_standard_error,
        },
    ]


# %%

# 0. import data from causalbench-datapreprocessing

# 1. want [Y, X1, ..., X10, E] where X1, ..., X10 are selected with feature importance on the observational data in decreasing order

# 2. [Y, X1, E = {obs, X1}] -> split train/test/test_same -> compare models
#    [Y, X1, X2, E = {obs, X1}]
#           .
#           .
#           .
#    [Y, X1, ..., X10, E = {obs, X1}]
#
#  [Y, X1, X2 E = {obs, X1, X2}]
#  [Y, X1, X2, X3, E = {obs, X1, X2}]
#           .
#           .
#           .
#  [Y, X1, ..., X10, E = {obs, X1, X2}]
#
#           .
#           .
#           .


# load data
path_train = "../../data/processed/train_genes.csv"
path_test = "../../data/processed/test_genes.csv"

X_train, y_train, Z_train = load_data(path_train)
X_test, y_test, Z_test = load_data(path_test)

MASK = np.repeat(True, 2)

# encode envs
Z_train_, Z_test_ = onehotencode(Z_train, Z_test)

mse_training = []
mse_test = []

# %%
#  fit models on training
bcf, bcf_lin, ls, ols = define_fit_models(X_train, y_train, Z_train_)


# %%
# predict models on training
f_hats = predict_models(bcf, bcf_lin, ls, ols, X_train, Z_train_)
if SAVE_PREDICTIONS:
    pd.concat([y_train, X_train, Z_train, f_hats], axis=1).to_csv(  # type: ignore
        f"{RESULT_PATH}genes-train_data{'_noLatLon' if ONLY_X_AT_PREDICTION else ''}.csv",
        index=False,
    )

# evaluate models on training
print("=================")
print(f"Train {0}")
mse_training += eval_models(
    bcf, bcf_lin, ls, ols, X_train, y_train, Z_train_, y_train, 0
)
print("=================")


# %%
# predict models on test
f_hats = predict_models(bcf, bcf_lin, ls, ols, X_test, Z_test_)
if SAVE_PREDICTIONS:
    pd.concat([y_test, X_test, Z_test, f_hats], axis=1).to_csv(
        f"{RESULT_PATH}genes-test_data_{0}{'_noLatLon' if ONLY_X_AT_PREDICTION else ''}.csv",
        index=False,
    )

# evaluate models on test
print("=================")
print(f"Test {0}")
mse_test += eval_models(bcf, bcf_lin, ls, ols, X_test, y_test, Z_test_, y_train, 0)
print("=================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BCF for a genes dataset.")

    # Set up logging
    # ...

# %%
