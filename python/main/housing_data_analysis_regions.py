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
from src.bcf.boosted_control_function_2 import BCF, OLS
from xgboost import XGBRegressor

# Constants and function definitions
DATA_PATH = "../data/processed/"
RESULT_PATH = "../results/output_data/"
SAVE_PREDICTIONS = False
NO_LAT_LON = True  # Set to True to use only X, and False to use [X, Z]
USE_WHOLE_DATA = False  # Set to True if don't split train/test
SPLIT_CRITERIONS = [
    "North/South",
    "South/North",
    "East/West",
    "West/East",
    "SE/rest",
    "rest/SE",
    "SW/rest",
    "rest/SW",
    "NE/rest",
    "rest/NE",
    "NW/rest",
    "rest/NW",
]
MASK = np.repeat(True, 7)


def load_data():
    dat = pd.read_csv(filepath_or_buffer=f"{DATA_PATH}housing-temp.csv")

    y = dat["MedHouseVal"]
    Z = dat[["Latitude", "Longitude"]]
    X = dat.drop(["MedHouseVal", "Latitude", "Longitude"], axis=1)

    return X, y, Z


def custom_train_test_split(
    X, y, Z, split_crit
) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame
]:
    # Calculate the median latitude and longitude for the splitting criteria
    median_lat = 35
    median_lon = -120

    conditions = {
        "South/North": lambda z: z["Latitude"] < median_lat,
        "North/South": lambda z: z["Latitude"] >= median_lat,
        "East/West": lambda z: z["Longitude"] >= median_lon,
        "West/East": lambda z: z["Longitude"] < median_lon,
        "SE/rest": lambda z: (z["Latitude"] < median_lat)
        & (z["Longitude"] >= median_lon),
        "rest/SE": lambda z: ~(
            (z["Latitude"] < median_lat) & (z["Longitude"] >= median_lon)
        ),
        "SW/rest": lambda z: (z["Latitude"] < median_lat)
        & (z["Longitude"] < median_lon),
        "rest/SW": lambda z: ~(
            (z["Latitude"] < median_lat) & (z["Longitude"] < median_lon)
        ),
        "NE/rest": lambda z: (z["Latitude"] >= median_lat)
        & (z["Longitude"] >= median_lon),
        "rest/NE": lambda z: ~(
            (z["Latitude"] >= median_lat) & (z["Longitude"] >= median_lon)
        ),
        "NW/rest": lambda z: (z["Latitude"] >= median_lat)
        & (z["Longitude"] < median_lon),
        "rest/NW": lambda z: ~(
            (z["Latitude"] >= median_lat) & (z["Longitude"] < median_lon)
        ),
    }

    if split_crit not in conditions:
        raise ValueError(f"Splitting criterion '{split_crit}' is not valid.")

    cond = conditions[split_crit](Z)
    cond_test = ~cond

    # Split data based on the condition
    train_idx = X[cond].index
    test_idx = X[cond_test].index

    X_train, y_train, Z_train = X.loc[train_idx], y.loc[train_idx], Z.loc[train_idx]
    X_test, y_test, Z_test = X.loc[test_idx], y.loc[test_idx], Z.loc[test_idx]

    return X_train, y_train, Z_train, X_test, y_test, Z_test


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
        fx=XGBRegressor(learning_rate=0.025),
        gv=XGBRegressor(learning_rate=0.05),
        fx_imp=XGBRegressor(learning_rate=0.05),
        passes=10,
    )

    bcf.fit(np.hstack([X_train, Z_train]), y_train.ravel())

    print(f"Rank M_0: {bcf.q_opt_}")

    # BCF-lin
    bcf_lin = BCF(
        n_exog=Z_train.shape[1],
        continuous_mask=MASK,
        fx=Ridge(),
        gv=Ridge(),
        fx_imp=XGBRegressor(learning_rate=0.05),
        passes=10,
    )

    bcf_lin.fit(np.hstack([X_train, Z_train]), y_train.ravel())

    # Input covariates for other models
    input_covariates = (
        X_train.to_numpy() if NO_LAT_LON else np.hstack([X_train, Z_train])
    )

    # LS
    ls = OLS(fx=XGBRegressor(learning_rate=0.05))

    ls.fit(input_covariates, y_train.ravel())

    # OLS
    ols = OLS(fx=Ridge())
    ols.fit(input_covariates, y_train.ravel())

    # Ave

    return bcf, bcf_lin, ls, ols


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
    input_covariates = X.to_numpy() if NO_LAT_LON else np.hstack([X, Z])

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
    input_covariates = X if NO_LAT_LON else np.hstack([X, Z])

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


def main():
    # load data
    X, y, Z = load_data()

    mse_training = []
    mse_test = []

    for i, split_crit in enumerate(SPLIT_CRITERIONS):
        print(f"Split {split_crit} -- {i + 1} out of {len(SPLIT_CRITERIONS)}")
        X_train, y_train, Z_train, X_test, y_test, Z_test = custom_train_test_split(
            X, y, Z, split_crit
        )

        print(f"Train: {X_train.shape[0]}")
        print(f"Test: {X_test.shape[0]}")

        # fit models on training
        bcf, bcf_lin, ls, ols = define_fit_models(X_train, y_train, Z_train)

        # predict models on training
        f_hats = predict_models(bcf, bcf_lin, ls, ols, X_train, Z_train)
        if SAVE_PREDICTIONS:
            pd.concat([y_train, X_train, Z_train, f_hats], axis=1).to_csv(  # type: ignore
                f"{RESULT_PATH}train_data_{i}{'_noLatLon' if NO_LAT_LON else ''}.csv",
                index=False,
            )

        # evaluate models on training
        print("=================")
        print(f"Train {split_crit}")
        mse_training += eval_models(
            bcf, bcf_lin, ls, ols, X_train, y_train, Z_train, y_train, split_crit
        )
        print("=================")

        # predict models on test
        f_hats = predict_models(bcf, bcf_lin, ls, ols, X_test, Z_test)
        if SAVE_PREDICTIONS:
            pd.concat([y_test, X_test, Z_test, f_hats], axis=1).to_csv(
                f"{RESULT_PATH}test_data_{i}{'_noLatLon' if NO_LAT_LON else ''}.csv",
                index=False,
            )

        # evaluate models on test
        print("=================")
        print(f"Test {split_crit}")
        mse_test += eval_models(
            bcf, bcf_lin, ls, ols, X_test, y_test, Z_test, y_train, split_crit
        )
        print("=================")

    pd.DataFrame(mse_training).to_csv(f"{RESULT_PATH}training_mse.csv", index=False)
    pd.DataFrame(mse_test).to_csv(f"{RESULT_PATH}test_mse.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BCF for a given chunk on the California housing dataset."
    )

    # Set up logging

    main()
