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
from sklearn.model_selection import GridSearchCV, train_test_split
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

# Define the number of repetitions
B = 1  # You can adjust this value as needed

# Define the validation as percentage training samples
VAL_PRECENTAGE = 0.2

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
    """Load and preprocess the California housing data."""
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
    """
    Split the data into training and testing sets based on geographic criteria.

    Parameters:
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    Z : pd.DataFrame
        Additional covariates (e.g., Latitude and Longitude).
    split_crit : str
        The splitting criterion.

    Returns:
    -------
    X_train : pd.DataFrame
    y_train : pd.Series
    Z_train : pd.DataFrame
    X_test : pd.DataFrame
    y_test : pd.Series
    Z_test : pd.DataFrame
    """
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


def fit_model(
    model: dict, X_train: pd.DataFrame, y_train: pd.Series, Z_train: pd.DataFrame
):
    """
    Fit a single model on the training data.

    Parameters:
    ----------
    model : dict
        A dictionary containing the model name and instance.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    Z_train : pd.DataFrame
        Additional covariates for models that require them.
    """
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
    """
    Generate predictions using a single model.

    Parameters:
    ----------
    model : dict
        A dictionary containing the model name and instance.
    X : pd.DataFrame
        Features for prediction.
    Z : pd.DataFrame
        Additional covariates for models that require them.

    Returns:
    -------
    pd.Series
        Predictions with the same index as X.
    """
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
    rep: str,
) -> dict:
    """
    Evaluate a single model and return its MSE and standard error.

    Parameters:
    ----------
    model : dict
        A dictionary containing the model name and instance.
    X : pd.DataFrame
        Features for evaluation.
    y : pd.Series
        True target values.
    Z : pd.DataFrame
        Additional covariates for models that require them.
    rep : str
        Representation identifier for the repetition.

    Returns:
    -------
    dict
        A dictionary containing the evaluation metrics.
    """
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
    }


def main():
    """
    Main function to execute the modeling and evaluation pipeline.
    """
    # Load data
    X, y, Z = load_data()

    mse_training = []
    mse_test = []

    for split_idx, split_crit in enumerate(SPLIT_CRITERIONS):
        print(
            f"\nProcessing Split: {split_crit} -- {split_idx + 1} out of {len(SPLIT_CRITERIONS)}"
        )
        X_train, y_train, Z_train, X_test, y_test, Z_test = custom_train_test_split(
            X, y, Z, split_crit
        )

        print(f"Training samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")

        for b in range(1, B + 1):
            print(f"\n--- Repetition {b} for Split '{split_crit}' ---")

            # Subsample X_train to create X_train_1 and X_valid
            X_train_1, X_valid, y_train_1, y_valid, Z_train_1, Z_valid = (
                train_test_split(
                    X_train,
                    y_train,
                    Z_train,
                    test_size=VAL_PRECENTAGE,
                    random_state=b,  # Ensures reproducibility for each repetition
                    shuffle=True,
                )
            )

            print(
                f"Subsample {b}: Training_1 samples: {X_train_1.shape[0]}, Validation samples: {X_valid.shape[0]}"
            )

            for model in MODELS:
                model_name = model["name"]
                rep_id = f"{split_crit}_Rep{b}"
                print(f"\nFitting Model: {model_name} (Repetition: {rep_id})")
                fit_model(model, X_train_1, y_train_1, Z_train_1)

                # Predict on Validation Data
                print(f"Predicting with Model: {model_name} on Validation Data")
                y_pred_valid = predict_model(model, X_valid, Z_valid)

                if SAVE_PREDICTIONS:
                    train_predictions = pd.concat(
                        [y_train_1, X_train_1, Z_train_1, y_pred_valid], axis=1
                    )
                    train_predictions.to_csv(
                        f"{RESULT_PATH}train_data_{split_idx}_{model_name}_Rep{b}{'_noLatLon' if NO_LAT_LON else ''}.csv",
                        index=False,
                    )

                # Evaluate on Validation Data
                print(f"Evaluating Model: {model_name} on Validation Data")
                eval_valid = evaluate_model(model, X_valid, y_valid, Z_valid, rep_id)
                mse_training.append(eval_valid)

                # Predict on Testing Data
                print(f"Predicting with Model: {model_name} on Testing Data")
                y_pred_test = predict_model(model, X_test, Z_test)

                if SAVE_PREDICTIONS:
                    test_predictions = pd.concat(
                        [y_test, X_test, Z_test, y_pred_test], axis=1
                    )
                    test_predictions.to_csv(
                        f"{RESULT_PATH}test_data_{split_idx}_{model_name}_Rep{b}{'_noLatLon' if NO_LAT_LON else ''}.csv",
                        index=False,
                    )

                # Evaluate on Testing Data
                print(f"Evaluating Model: {model_name} on Testing Data")
                eval_test = evaluate_model(model, X_test, y_test, Z_test, rep_id)
                mse_test.append(eval_test)

    # Save evaluation results
    training_mse_df = pd.DataFrame(mse_training)
    test_mse_df = pd.DataFrame(mse_test)

    training_mse_output_path = os.path.join(RESULT_PATH, "training_mse-new.csv")
    test_mse_output_path = os.path.join(RESULT_PATH, "test_mse-new.csv")

    training_mse_df.to_csv(training_mse_output_path, index=False)
    test_mse_df.to_csv(test_mse_output_path, index=False)

    print("\nEvaluation complete. Results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BCF for a given chunk on the California housing dataset."
    )

    # Set up logging (optional)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            # You can add FileHandler here to log to a file
        ],
    )

    args = parser.parse_args()

    main()
