# %%
import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, TypedDict, Union

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from src.bcf.boosted_control_function_2 import BCF, OLS, MeanModel
from src.bcf.boosted_control_function_mlp import BCFMLP, OLSMLP
from src.bcf.mlp import MLP
from xgboost import XGBRegressor


class ModelItem(TypedDict):
    name: str
    instance: Union[BCF, BCFMLP, OLS, MeanModel, OLSMLP]


ListOfModels = List[ModelItem]


# %%
def make_mlp_sigmoid(in_dim: int):
    return MLP(in_dim=in_dim, hidden=[64], activation=nn.Sigmoid)  # type: ignore


def get_models() -> ListOfModels:
    """
    Returns a fresh list of model dictionaries with new instances.
    """
    MASK = np.repeat(True, 7)  # Define the continuous mask here if it's dynamic
    models: List[ModelItem] = [
        {
            "name": "BCF",
            "instance": BCF(
                n_exog=2,  # Latitude and Longitude
                continuous_mask=MASK,
                fx=XGBRegressor(learning_rate=0.05, base_score=0.0),
                gv=XGBRegressor(learning_rate=0.05, base_score=0.0),
                fx_imp=XGBRegressor(learning_rate=0.05, base_score=0.0),
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
                fx_imp=XGBRegressor(learning_rate=0.05, base_score=0.0),
                passes=10,
            ),
        },
        {
            "name": "LS",
            "instance": OLS(fx=XGBRegressor(learning_rate=0.05, base_score=0.0)),
        },
        {"name": "OLS", "instance": OLS(fx=Ridge())},
        {"name": "AVE", "instance": MeanModel()},
        {
            "name": "CF-medium",
            "instance": BCF(
                n_exog=2,  # Latitude and Longitude
                continuous_mask=MASK,
                fx=XGBRegressor(learning_rate=0.05, base_score=0.0),
                gv=XGBRegressor(learning_rate=0.05, base_score=0.0),
                fx_imp=XGBRegressor(learning_rate=0.05, base_score=0.0),
                passes=10,
                predict_imp=False,
            ),
        },
        {
            "name": "CF-small",
            "instance": BCF(
                n_exog=2,  # Latitude and Longitude
                continuous_mask=MASK,
                fx=XGBRegressor(learning_rate=0.025, base_score=0.0),
                gv=XGBRegressor(learning_rate=0.025, base_score=0.0),
                fx_imp=XGBRegressor(learning_rate=0.05, base_score=0.0),
                passes=10,
                predict_imp=False,
            ),
        },
        {
            "name": "CF-large",
            "instance": BCF(
                n_exog=2,  # Latitude and Longitude
                continuous_mask=MASK,
                fx=XGBRegressor(learning_rate=0.10, base_score=0.0),
                gv=XGBRegressor(learning_rate=0.10, base_score=0.0),
                fx_imp=XGBRegressor(learning_rate=0.05, base_score=0.0),
                passes=10,
                predict_imp=False,
            ),
        },
        {
            "name": "BCF-MLP",
            "instance": BCFMLP(
                n_exog=2,  # Latitude and Longitude
                continuous_mask=MASK,
                fx_factory=make_mlp_sigmoid,
                fx_imp_factory=make_mlp_sigmoid,
                gv_factory=make_mlp_sigmoid,
                epochs_step_1=1000,
                epochs_step_2=1500,
                lr_step_1=1e-3,
                lr_step_2=1e-3,
                weight_decay_step_1=2.5e-3,
                weight_decay_step_2=0.0,
            ),
        },
        {
            "name": "CF-MLP",
            "instance": BCFMLP(
                n_exog=2,  # Latitude and Longitude
                continuous_mask=MASK,
                fx_factory=make_mlp_sigmoid,
                fx_imp_factory=make_mlp_sigmoid,
                gv_factory=make_mlp_sigmoid,
                epochs_step_1=1000,
                lr_step_1=1e-3,
                weight_decay_step_1=2.5e-1,
                predict_imp=False,
            ),
        },
        {
            "name": "OLS-MLP",
            "instance": OLSMLP(
                continuous_mask=MASK,
                fx_factory=make_mlp_sigmoid,
                epochs=1000,
                lr=1e-3,
                weight_decay=2.5e-3,
            ),
        },
    ]

    return models


def filter_models(
    models: ListOfModels,
    models_selected: List[str],
) -> ListOfModels:

    return [
        model_dict for model_dict in models if model_dict["name"] in models_selected
    ]


def load_data(data_path: str):
    """Load and preprocess the California housing data."""
    dat = pd.read_csv(filepath_or_buffer=f"{data_path}/housing-temp.csv")

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
    model: ModelItem,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    Z_train: pd.DataFrame,
    no_lat_lon: bool,
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

    if isinstance(model_instance, BCF) or isinstance(model_instance, BCFMLP):
        # BCF models require combined X and Z
        input_data = np.hstack([X_train, Z_train])
        model_instance.fit(input_data, y_train.ravel())
        print(f"Model {model_name} fitted. Rank M_0: {model_instance.q_opt_}")
    else:
        # Other models use X or [X, Z] based on NO_LAT_LON
        input_covariates = (
            X_train.to_numpy() if no_lat_lon else np.hstack([X_train, Z_train])
        )
        model_instance.fit(input_covariates, y_train.ravel())
        print(f"Model {model_name} fitted.")


def predict_model(
    model: ModelItem, X: pd.DataFrame, Z: pd.DataFrame, no_lat_lon: bool
) -> pd.Series:
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

    if isinstance(model_instance, BCF) or isinstance(model_instance, BCFMLP):
        f_bcf = model_instance.predict(X.to_numpy())
        return pd.Series(f_bcf, index=X.index, name=model_name)
    else:
        input_covariates = X.to_numpy() if no_lat_lon else np.hstack([X, Z])
        predictions = model_instance.predict(input_covariates)
        return pd.Series(predictions, index=X.index, name=model_name)


def evaluate_model(
    model: ModelItem,
    X: pd.DataFrame,
    y: pd.Series,
    Z: pd.DataFrame,
    rep: str,
    no_lat_lon: bool,
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

    if isinstance(model_instance, BCF) or isinstance(model_instance, BCFMLP):
        y_pred = model_instance.predict(np.hstack([X, Z]))
    else:
        input_covariates = X.to_numpy() if no_lat_lon else np.hstack([X, Z])
        y_pred = model_instance.predict(input_covariates)

    mse = ((y_pred - y) ** 2).mean()
    se = ((y_pred - y) ** 2).std() / np.sqrt(len(y))

    return {
        "Rep": rep,
        "Method": model_name,
        "MSE": mse,
        "MSE_standard_error": se,
    }


def process_repetition(
    *,
    split_idx: int,
    split_crit: str,
    b: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    Z_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    Z_test: pd.DataFrame,
    val_percentage: float,
    models_selected: list[str],
    no_lat_lon: bool,
    save_predictions: bool,
) -> Tuple[List[dict], List[dict]]:
    """
    Process a single repetition for a given split criterion.

    Parameters:
    ----------
    split_idx : int
        Index of the split criterion.
    split_crit : str
        The splitting criterion.
    b : int
        Repetition number.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    Z_train : pd.DataFrame
        Training additional covariates.
    X_test : pd.DataFrame
        Testing features.
    y_test : pd.Series
        Testing target.
    Z_test : pd.DataFrame
        Testing additional covariates.
    val_percentage : float, optional
        Percentage of training data to use for validation.

    Returns:
    -------
    Tuple[List[dict], List[dict]]
        A tuple containing lists of evaluation dictionaries for validation and testing.
    """
    # Limit per-process threading to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # Subsample the training data into train_1 and validation sets
    X_train_1, X_valid, y_train_1, y_valid, Z_train_1, Z_valid = train_test_split(
        X_train,
        y_train,
        Z_train,
        test_size=val_percentage,
        random_state=b,  # Ensures reproducibility for each repetition
        shuffle=True,
    )

    rep_id = f"{split_crit}_Rep{b}"
    print(f"\n--- Repetition {b} for Split '{split_crit}' ---")
    print(
        f"Subsample {b}: Training_1 samples: {X_train_1.shape[0]}, Validation samples: {X_valid.shape[0]}"
    )

    # Initialize fresh models
    models = filter_models(get_models(), models_selected=models_selected)

    mse_validation = []
    mse_testing = []

    for model in models:
        model_name = model["name"]
        print(f"\nFitting Model: {model_name} (Repetition: {rep_id})")
        fit_model(model, X_train_1, y_train_1, Z_train_1, no_lat_lon)

        # Predict on Validation Data
        print(f"Predicting with Model: {model_name} on Validation Data")
        y_pred_valid = predict_model(model, X_valid, Z_valid, no_lat_lon)

        if save_predictions:
            train_predictions = pd.concat(
                [
                    y_train_1.reset_index(drop=True),
                    X_train_1.reset_index(drop=True),
                    Z_train_1.reset_index(drop=True),
                    y_pred_valid.reset_index(drop=True),
                ],
                axis=1,
            )
            train_predictions.to_csv(
                f"train_data_{split_idx}_{model_name}_Rep{b}{'_noLatLon' if no_lat_lon else ''}.csv",
                index=False,
            )

        # Evaluate on Validation Data
        print(f"Evaluating Model: {model_name} on Validation Data")
        eval_valid = evaluate_model(
            model, X_valid, y_valid, Z_valid, rep_id, no_lat_lon
        )
        mse_validation.append(eval_valid)

        # Predict on Testing Data
        print(f"Predicting with Model: {model_name} on Testing Data")
        y_pred_test = predict_model(model, X_test, Z_test, no_lat_lon)

        if save_predictions:
            test_predictions = pd.concat(
                [
                    y_test.reset_index(drop=True),
                    X_test.reset_index(drop=True),
                    Z_test.reset_index(drop=True),
                    y_pred_test.reset_index(drop=True),
                ],
                axis=1,
            )
            test_predictions.to_csv(
                f"test_data_{split_idx}_{model_name}_Rep{b}{'_noLatLon' if no_lat_lon else ''}.csv",
                index=False,
            )

        # Evaluate on Testing Data
        print(f"Evaluating Model: {model_name} on Testing Data")
        eval_test = evaluate_model(model, X_test, y_test, Z_test, rep_id, no_lat_lon)
        mse_testing.append(eval_test)

    return mse_validation, mse_testing


@hydra.main(
    config_path="../configs/housing-data",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig):
    """
    Main function to execute the modeling and evaluation pipeline with parallelization.
    """

    save_predictions = bool(cfg.save_predictions)
    no_lat_lon = bool(cfg.no_lat_lon)
    val_percentage = float(cfg.val_percentage)
    models_selected = list(cfg.models_selected)

    # Load data
    old_path = Path(get_original_cwd()) / cfg.data_path
    X, y, Z = load_data(old_path)

    # Initialize lists to store evaluation results
    mse_validation_all, mse_testing_all = [], []

    # Iterate over each split criterion
    for split_idx, split_crit in enumerate(cfg.split_criterions):
        print(
            f"\nProcessing Split: {split_crit} -- {split_idx + 1} out of {len(cfg.split_criterions)}"
        )
        X_train, y_train, Z_train, X_test, y_test, Z_test = custom_train_test_split(
            X, y, Z, split_crit
        )

        print(f"Training samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")

        # Prepare a list of repetition indices
        repetition_indices = list(range(1, cfg.B + 1))

        # Define the list of tasks for this split criterion
        tasks = [(split_idx, split_crit, b) for b in repetition_indices]

        max_workers = (
            os.cpu_count() if cfg.num_workers is None else int(cfg.num_workers)
        )

        if cfg.debug:
            process_repetition(
                split_idx=split_idx,
                split_crit=split_crit,
                b=cfg.B,
                X_train=X_train,
                y_train=y_train,
                Z_train=Z_train,
                X_test=X_test,
                y_test=y_test,
                Z_test=Z_test,
                val_percentage=val_percentage,
                models_selected=models_selected,
                no_lat_lon=no_lat_lon,
                save_predictions=save_predictions,
            )
        # Execute the repetitions in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_task = {
                executor.submit(
                    process_repetition,
                    split_idx=split_idx,
                    split_crit=split_crit,
                    b=b,
                    X_train=X_train,
                    y_train=y_train,
                    Z_train=Z_train,
                    X_test=X_test,
                    y_test=y_test,
                    Z_test=Z_test,
                    val_percentage=val_percentage,
                    models_selected=models_selected,
                    no_lat_lon=no_lat_lon,
                    save_predictions=save_predictions,
                ): (split_idx, split_crit, b)
                for (split_idx, split_crit, b) in tasks
            }

            # As each task completes, collect the results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    mse_valid, mse_test = future.result()
                    mse_validation_all.extend(mse_valid)
                    mse_testing_all.extend(mse_test)
                except Exception as exc:
                    print(f"Task {task} generated an exception: {exc}")

    # Save evaluation results
    training_mse_df = pd.DataFrame(mse_validation_all)
    testing_mse_df = pd.DataFrame(mse_testing_all)

    training_mse_output_path = os.path.join("training_mse.csv")
    testing_mse_output_path = os.path.join("test_mse.csv")

    training_mse_df.to_csv("training_mse.csv", index=False)
    testing_mse_df.to_csv("test_mse-new.csv", index=False)

    print("\nEvaluation complete. Results saved.")


if __name__ == "__main__":
    main()
