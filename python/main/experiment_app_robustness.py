# %%
import hashlib
from pathlib import Path
from typing import List, Literal, Union

import hydra
import numpy as np
import pandas as pd
import torch.nn as nn
from numpy.random import SeedSequence
from omegaconf import DictConfig
from sklearn.linear_model import LinearRegression
from src.algorithms.oracle_methods import ConstantFunc, IMPFunctionNonLin
from src.bcf.boosted_control_function_2 import BCF, OLS
from src.bcf.boosted_control_function_mlp import BCFMLP
from src.bcf.mlp import MLP
from src.scenarios.generate_data import generate_data_radial_f
from src.scenarios.generate_helpers import radial2D
from src.simulations.simulations_funcs import compute_mse
from xgboost import XGBRegressor

LR = 0.25


#  ---- Function definitions
def factory_bcf_mlp(n_exog, continuous_mask, hidden):

    fx_factory = lambda x: MLP(in_dim=x, hidden=hidden, activation=nn.Sigmoid)  # type: ignore

    return BCFMLP(
        n_exog=n_exog,
        continuous_mask=continuous_mask,
        fx_factory=fx_factory,
        fx_imp_factory=fx_factory,
        gv_factory=fx_factory,
        epochs_step_1=1000,
        epochs_step_2=1500,
        lr_f=1e-3,
        lr_g=1e-3,
        lr_fimp=1e-4,
        weight_decay_f=2.5e-3,
        weight_decay_g=2.5e-3,
        weight_decay_fimp=0.0,
    )


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
    selected_methods,
    n_train,
    n_test,
    num_basis,
    g,
    is_nonlinear_g,
    instrument_strength,
    is_instrument_discrete,
    noise_sd,
    noise_sd_Y,
    int_train,
    ints_test,
    rng_numpy,
    seed_torch,
):
    # Sample random function f
    f = radial2D(num_basis=num_basis, seed=rng_numpy)

    #  generate training data
    X_train, y_train, Z_train, S, M, g_scaled = generate_data_radial_f(
        n_train,
        int_train,
        f,
        g,
        instrument_strength=instrument_strength,
        instrument_discrete=is_instrument_discrete,
        noise_sd=noise_sd,
        noise_sd_Y=noise_sd_Y,
        scale_g=False,
        seed=rng_numpy,
    )

    # set up methods
    # Define methods

    methods = [
        (
            "BCF",
            BCF(
                n_exog=Z_train.shape[1],
                continuous_mask=np.repeat(True, X_train.shape[1]),
                passes=75,
                fx=XGBRegressor(learning_rate=0.05),
                gv=XGBRegressor(learning_rate=0.05),
                fx_imp=XGBRegressor(learning_rate=LR),
                predict_imp=True,
                tol_delta=5e-3,
            ),
        ),
        (
            "BCF-MLP-small",
            factory_bcf_mlp(
                n_exog=Z_train.shape[1],
                continuous_mask=np.repeat(True, X_train.shape[1]),
                hidden=[16],
            ),
        ),
        (
            "BCF-MLP-medium",
            factory_bcf_mlp(
                n_exog=Z_train.shape[1],
                continuous_mask=np.repeat(True, X_train.shape[1]),
                hidden=[32],
            ),
        ),
        (
            "BCF-MLP-large",
            factory_bcf_mlp(
                n_exog=Z_train.shape[1],
                continuous_mask=np.repeat(True, X_train.shape[1]),
                hidden=[64],
            ),
        ),
        (
            "IMP",
            IMPFunctionNonLin(
                causal_function=f,
                instrument_matrix=M,
                n_exog=Z_train.shape[1],
                confounder_cov=S,
                confounder_effect=g_scaled,
                mode="exact",
            ),
        ),
        (
            "Causal",
            IMPFunctionNonLin(
                causal_function=f,
                instrument_matrix=M,
                n_exog=Z_train.shape[1],
                confounder_cov=S,
                confounder_effect=g_scaled,
                mode="exact",
                use_imp=False,
            ),
        ),
        (
            "OLS",
            OLS(fx=XGBRegressor(learning_rate=LR)),
        ),
    ]

    # generate test datasets
    test_datasets = [
        generate_data_radial_f(
            n_test,
            int_par,
            f,
            g_scaled,
            instrument_strength=instrument_strength,
            instrument_discrete=False,
            noise_sd=noise_sd,
            noise_sd_Y=noise_sd_Y,
            scale_g=False,
            seed=seed_from_string(f"robustness-int={int_par}"),
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

    methods_ = filter_methods(methods, selected_methods)
    # fit/predict/evaluate methods
    for method_name, method in methods_:
        # fit methods
        print(method_name)

        expected_classes = (IMPFunctionNonLin, BCF, BCFMLP, ConstantFunc, OLS)

        if isinstance(method, IMPFunctionNonLin):
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
            int_par=int_train,
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
            var_y = np.var(y)

            mses = pd.concat(
                [
                    mses,
                    pd.DataFrame.from_dict(
                        {
                            "model": [method_name],
                            "test_mse": [test_mse],
                            "int_par": [int_par],
                            "var_y": [var_y],
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
    config_path="../configs/exp-robustness",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    # Set seeds
    base_seed = seed_from_string(f"robustness-experiment-rep_id={cfg.rep_id}")

    ss = SeedSequence(base_seed)
    children = ss.spawn(cfg.n_reps * 2)  # child sequences

    # RNGs for generating data
    rng_numpy_children = children[: cfg.n_reps]  # child sequences

    # Seeds for each algorithms using torch, if applicable
    seeds_torch = [
        int(c.generate_state(1)[0]) for c in children[cfg.n_reps :]
    ]  # python ints

    # set up test intervention strengths
    intvec = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 3.99]

    # Generate function g
    if cfg.nonlinear_g:

        def g_(v):
            return np.tanh(v[:, 0] + v[:, 1])

        rng = np.random.default_rng(42)
        v = rng.normal(size=(10000, 2))
        mu, sigma = np.mean(g_(v)), np.std(g_(v))
        g = lambda v: (g_(v) - mu) / sigma * np.sqrt(2)
    else:
        g = lambda v: v[:, 0] + v[:, 1]

    # iterate simulations over n_reps
    all_methods = []
    all_mses = []

    for b in range(cfg.n_reps):
        print(f"Iteration {b+1} out of {cfg.n_reps}")
        res_methods_, res_mses_ = simulation_run(
            selected_methods=cfg.selected_methods,
            n_train=cfg.n_train,
            n_test=cfg.n_test,
            num_basis=cfg.num_basis,
            g=g,
            is_nonlinear_g=cfg.nonlinear_g,
            instrument_strength=cfg.instrument_strength,
            is_instrument_discrete=cfg.instrument_discrete,
            noise_sd=cfg.noise_sd,
            noise_sd_Y=cfg.noise_sd_Y,
            int_train=cfg.int_train,
            ints_test=intvec,
            rng_numpy=rng_numpy_children[b],
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
