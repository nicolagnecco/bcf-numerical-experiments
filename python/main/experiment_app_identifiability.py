# %%
import hashlib
from pathlib import Path
from typing import List, Literal, Union

import hydra
import numpy as np
import pandas as pd
import torch.nn as nn
from hydra.utils import get_original_cwd
from numpy.random import SeedSequence
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LinearRegression
from src.algorithms.oracle_methods import ConstantFunc, IMPFunctionNonLin
from src.bcf.boosted_control_function_2 import BCF, OLS
from src.bcf.boosted_control_function_mlp import BCFMLP, OLSMLP
from src.bcf.mlp import MLP
from src.scenarios.generate_data import generate_data_radial_f
from src.scenarios.generate_helpers import radial2D
from src.simulations.simulations_funcs import compute_mse
from xgboost import XGBRegressor


#  ---- Function definitions
def seed_from_string(s: str, offset: int = 0) -> int:
    h = hashlib.sha1(s.encode()).digest()[:4]  # 32-bit
    return (int.from_bytes(h, "big") + offset) & 0x7FFFFFFF  # < 2^31


def create_dataframe(
    X: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
    env: str,
    model: str,
    int_par: float,
    col_names: List[str],
) -> pd.DataFrame:
    dat = pd.DataFrame(
        data=np.concatenate([X, y[:, np.newaxis], y_hat[:, np.newaxis]], axis=1),
        columns=col_names,
    )
    dat["model"] = model
    dat["env"] = env
    dat["int_par"] = int_par
    return dat


def simulation_run(
    *,
    methods,
    n_train,
    n_test,
    num_basis,
    g,
    instrument_strength,
    is_instrument_discrete,
    noise_sd,
    int_train,
    ints_test,
    rng_numpy,
    seed_torch,
):
    # Sample random function f
    f = radial2D(num_basis=num_basis, seed=rng_numpy)

    #  generate datasets
    X_train, y_train, Z_train, _, _, _ = generate_data_radial_f(
        n_train,
        int_train,
        f,
        g,
        instrument_strength=instrument_strength,
        instrument_discrete=is_instrument_discrete,
        noise_sd=noise_sd,
        seed=rng_numpy,
    )

    test_datasets = [
        generate_data_radial_f(
            n_test,
            int_par,
            f,
            g,
            instrument_strength=instrument_strength,
            instrument_discrete=False,
            noise_sd=noise_sd,
            seed=rng_numpy,
        )
        for int_par in ints_test
    ]

    # preallocate dataframes
    mses = pd.DataFrame(columns=["model", "test_mse", "int_par"])
    col_names = ["X{j}".format(j=j + 1) for j in range(X_train.shape[1])] + [
        "y",
        "y_hat",
    ]
    dat_methods = pd.DataFrame(columns=col_names + ["model", "env", "int_par"])

    # fit/predict/evaluate methods
    for method_name, method in methods:
        # fit methods
        print(method_name)

        expected_classes = (IMPFunctionNonLin, BCF, BCFMLP, ConstantFunc, OLS)

        if isinstance(method, IMPFunctionNonLin):
            method.causal_function = f
            method.fit(np.hstack([X_train, Z_train]), y_train, seed=seed_torch)

        elif isinstance(method, (BCF, BCFMLP)):
            method.fit(np.hstack([X_train, Z_train]), y_train, seed=seed_torch)

        elif isinstance(method, ConstantFunc):
            method.fit(X_train, y_train, Z_train)

        elif isinstance(method, OLS):
            method.fit(X_train, y_train, seed=seed_torch)

        else:
            # Raise an informative error listing all accepted classes
            valid_class_names = [cls.__name__ for cls in expected_classes]
            raise TypeError(
                f"Unexpected method class: {method.__class__.__name__}. "
                f"Expected one of: {', '.join(valid_class_names)}."
            )

        # ---- predict on training data
        y_hat = method.predict(X_train)

        # save training data and predictions
        inner_dat = create_dataframe(
            X=X_train,
            y=y_train,
            y_hat=y_hat,
            env="train",
            model=method_name,
            int_par=0.5,
            col_names=col_names,
        )

        dat_methods = pd.concat([dat_methods, inner_dat], ignore_index=True)

        # generate test data
        for i, int_par in enumerate(ints_test):
            X, y, Z, _, _, _ = test_datasets[i]

            # ---- predict on test data and save
            y_hat = method.predict(X)

            inner_dat = create_dataframe(
                X=X,
                y=y,
                y_hat=y_hat,
                env="test",
                model=method_name,
                int_par=int_par,
                col_names=col_names,
            )

            dat_methods = pd.concat([dat_methods, inner_dat], ignore_index=True)

            test_mse = compute_mse(y_hat, y)

            mses = pd.concat(
                [
                    mses,
                    pd.DataFrame.from_dict(
                        {
                            "model": [method_name],
                            "test_mse": [test_mse],
                            "int_par": [int_par],
                        }
                    ),
                ],
                ignore_index=True,
            )

    return dat_methods, mses


def filter_methods(methods, selected_methods: Union[List[str], Literal["all"]]):
    if selected_methods == "all":
        return methods
    else:
        return [(a, b) for a, b in methods if a in selected_methods]


# %%
@hydra.main(
    config_path="../configs/exp-identifiability",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    # Set seeds
    base_seed = seed_from_string(cfg.run_name)

    # RNG for generating data
    rng_numpy = np.random.default_rng(base_seed)

    # Seeds for each algorithms using torch, if applicable
    ss = SeedSequence(base_seed)
    children = ss.spawn(cfg.n_reps)  # 10 child sequences
    seeds_torch = [int(c.generate_state(1)[0]) for c in children]  # 10 python ints

    #  Generate random function f and oracle quantities
    if cfg.instrument_discrete:
        intvec = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 3.99]
    else:
        intvec = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 3.99]

    f = radial2D(num_basis=cfg.num_basis, seed=rng_numpy)

    if cfg.nonlinear_g:
        g = (
            lambda v: np.sin(v[:, 0] * v[:, 1])
            + 8 * np.tanh(v[:, 0] / 8)
            + 8 * np.tanh(v[:, 1] / 8)
        )
        g = radial2D(num_basis=cfg.num_basis, seed=rng_numpy)
        scale_g = True
    else:
        g = lambda v: v[:, 0] + v[:, 1]
        scale_g = False

    X, _, Z, S, M, gamma = generate_data_radial_f(
        cfg.n_train,
        intvec[0],
        f,
        g=g,
        instrument_strength=cfg.instrument_strength,
        instrument_discrete=cfg.instrument_discrete,
        noise_sd=cfg.noise_sd,
        scale_g=scale_g,
        seed=rng_numpy,
    )

    # Define methods
    methods = [
        (
            "BCF",
            BCF(
                n_exog=Z.shape[1],
                continuous_mask=np.repeat(True, X.shape[1]),
                passes=10,
                fx=XGBRegressor(learning_rate=0.05),
                gv=(
                    XGBRegressor(learning_rate=0.05)
                    if cfg.nonlinear_g
                    else XGBRegressor(learning_rate=0.05)
                ),
                fx_imp=XGBRegressor(learning_rate=0.05),
            ),
        ),
        (
            "CF",
            BCF(
                n_exog=Z.shape[1],
                continuous_mask=np.repeat(True, X.shape[1]),
                passes=10,
                fx=XGBRegressor(learning_rate=0.05),
                gv=(
                    XGBRegressor(learning_rate=0.05)
                    if cfg.nonlinear_g
                    else XGBRegressor(learning_rate=0.05)
                ),
                fx_imp=XGBRegressor(learning_rate=0.05),
                predict_imp=False,
            ),
        ),
        (
            "OLS",
            OLS(fx=XGBRegressor(learning_rate=0.05)),
        ),
        (
            "IMP",
            IMPFunctionNonLin(
                causal_function=f,
                instrument_matrix=M,
                n_exog=Z.shape[1],
                confounder_cov=S,
                confounder_effect=gamma,
                mode="computed",
                boosted_estimator=XGBRegressor(learning_rate=0.05),
            ),
        ),
        (
            "Causal",
            IMPFunctionNonLin(
                causal_function=f,
                instrument_matrix=M,
                n_exog=Z.shape[1],
                confounder_cov=S,
                confounder_effect=gamma,
                mode="computed",
                boosted_estimator=XGBRegressor(learning_rate=0.05),
                use_imp=False,
            ),
        ),
    ]

    # iterate simulations over n_reps
    all_methods = []
    all_mses = []
    methods_ = filter_methods(methods, cfg.selected_methods)

    for b in range(cfg.n_reps):
        print(f"Iteration {b+1} out of {cfg.n_reps}")
        res_methods_, res_mses_ = simulation_run(
            methods=methods_,
            n_train=cfg.n_train,
            n_test=cfg.n_test,
            num_basis=cfg.num_basis,
            g=gamma,
            instrument_strength=cfg.instrument_strength,
            is_instrument_discrete=cfg.instrument_discrete,
            noise_sd=cfg.noise_sd,
            int_train=intvec[0],
            ints_test=intvec,
            rng_numpy=rng_numpy,
            seed_torch=seeds_torch[b],
        )

        res_methods_["rep_id"] = b
        res_mses_["rep_id"] = b
        all_methods.append(res_methods_)
        all_mses.append(res_mses_)

    # write csv
    pd.concat(all_methods, ignore_index=True).to_csv("predictions.csv", index=False)
    pd.concat(all_mses, ignore_index=True).to_csv("mses.csv", index=False)


if __name__ == "__main__":
    main()
