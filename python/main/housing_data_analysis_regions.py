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
from src.bcf.boosted_control_function_2 import BCF, OLS, MeanModel
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

# Define a list of models globally
MODELS = [
    {
        "name": "BCF",
        "instance": BCF(
            n_exog=2,  # Assuming Z_train has 2 features: Latitude and Longitude
            continuous_mask=MASK,
            fx=XGBRegressor(learning_rate=0.025),
            gv=XGBRegressor(learning_rate=0.05),
            fx_imp=XGBRegressor(learning_rate=0.05),
            passes=10,
        ),
    },
    {
        "name": "BCF-lin",
        "instance": BCF(
            n_exog=2,
            continuous_mask=MASK,
            fx=Ridge(),
            gv=Ridge(),
            fx_imp=XGBRegressor(learning_rate=0.05),
            passes=10,
        ),
    },
    {
        "name": "LS",
        "instance": OLS(fx=XGBRegressor(learning_rate=0.05)),
    },
    {
        "name": "OLS",
        "instance": OLS(fx=Ridge()),
    },
    {
        "name": "AVE",
        "instance": MeanModel(),
    },
]


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


def fit_model(
    model: dict, X_train: pd.DataFrame, y_train: pd.Series, Z_train: pd.DataFrame
):
    """Fit a single model on the training data."""
    model_instance = model["instance"]
    model_name = model["name"]

    if isinstance(model_instance, BCF):
        # BCF models require combined X and Z
        input_data = np.hstack([X_train, Z_train])
        model_instance.fit(input_data, y_train.ravel())
        print(f"Model {model_name} fitted. Rank M_0: {model_instance.q_opt_}")
    else:
        # Other models use X or [X, Z] based on NO_LAT_LON
        input_covariates = (
            X_train.to_numpy() if NO_LAT_LON else np.hstack([X_train, Z_train])
        )
        model_instance.fit(input_covariates, y_train.ravel())
        print(f"Model {model_name} fitted.")


def predict_model(model: dict, X: pd.DataFrame, Z: pd.DataFrame) -> pd.Series:
    """Generate predictions using a single model."""
    model_instance = model["instance"]
    model_name = model["name"]

    if isinstance(model_instance, BCF):
        f_bcf = model_instance.predict(X.to_numpy())
        return pd.Series(f_bcf, index=X.index, name=model_name)
    else:
        input_covariates = X.to_numpy() if NO_LAT_LON else np.hstack([X, Z])
        predictions = model_instance.predict(input_covariates)
        return pd.Series(predictions, index=X.index, name=model_name)


def evaluate_model(
    model: dict,
    X: pd.DataFrame,
    y: pd.Series,
    Z: pd.DataFrame,
    y_train: pd.Series,
    rep: str,
) -> dict:
    """Evaluate a single model and return its MSE and standard error."""
    model_instance = model["instance"]
    model_name = model["name"]

    if isinstance(model_instance, BCF):
        y_pred = model_instance.predict(np.hstack([X, Z]))
    else:
        input_covariates = X.to_numpy() if NO_LAT_LON else np.hstack([X, Z])
        y_pred = model_instance.predict(input_covariates)

    mse = ((y_pred - y) ** 2).mean()
    se = ((y_pred - y) ** 2).std() / np.sqrt(len(y))

    return {
        "Rep": rep,
        "Method": model_name,
        "MSE": mse,
        "MSE_standard_error": se,
    }


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

        for model in MODELS:
            model_name = model["name"]
            print(f"\nFitting Model: {model_name}")
            fit_model(model, X_train, y_train, Z_train)

            # Predict on Training Data
            y_pred_train = predict_model(model, X_train, Z_train)
            if SAVE_PREDICTIONS:
                train_predictions = pd.concat(
                    [y_train, X_train, Z_train, y_pred_train], axis=1
                )
                train_predictions.to_csv(
                    f"{RESULT_PATH}train_data_{i}_{model_name}{'_noLatLon' if NO_LAT_LON else ''}.csv",
                    index=False,
                )

            # Evaluate on Training Data
            print(f"Evaluating Model: {model_name} on Training Data")
            eval_train = evaluate_model(
                model, X_train, y_train, Z_train, y_train, split_crit
            )
            mse_training.append(eval_train)

            # Predict on Testing Data
            y_pred_test = predict_model(model, X_test, Z_test)
            if SAVE_PREDICTIONS:
                test_predictions = pd.concat(
                    [y_test, X_test, Z_test, y_pred_test], axis=1
                )
                test_predictions.to_csv(
                    f"{RESULT_PATH}test_data_{i}_{model_name}{'_noLatLon' if NO_LAT_LON else ''}.csv",
                    index=False,
                )

            # Evaluate on Testing Data
            print(f"Evaluating Model: {model_name} on Testing Data")
            eval_test = evaluate_model(
                model, X_test, y_test, Z_test, y_train, split_crit
            )
            mse_test.append(eval_test)

    # Save evaluation results
    pd.DataFrame(mse_training).to_csv(f"{RESULT_PATH}training_mse-new.csv", index=False)
    pd.DataFrame(mse_test).to_csv(f"{RESULT_PATH}test_mse-new.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BCF for a given chunk on the California housing dataset."
    )

    # Set up logging

    main()
