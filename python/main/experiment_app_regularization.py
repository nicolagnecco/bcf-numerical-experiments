# %%
from pathlib import Path
from typing import List, Literal, Union

import hydra
import numpy as np
import pandas as pd
import torch.nn as nn
from hydra.utils import get_original_cwd
from numpy.random import SeedSequence
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor
from src.algorithms.oracle_methods import ConstantFunc, IMPFunctionNonLin
from src.bcf.boosted_control_function_2 import BCF, OLS
from src.bcf.boosted_control_function_mlp import BCFMLP, OLSMLP
from src.bcf.mlp import MLP
from src.scenarios.generate_data import generate_data_radial_f
from src.scenarios.generate_helpers import radial2D
from src.simulations.simulations_funcs import compute_mse
from xgboost import XGBRegressor


#  ---- Function definitions
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
    methods,
    n_train,
    n_test,
    num_basis,
    instrument_strength,
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
        instrument_strength=instrument_strength,
        noise_sd=noise_sd,
        seed=rng_numpy,
    )
    Z_train = Z_train[:, np.newaxis]

    test_datasets = [
        generate_data_radial_f(
            n_test,
            int_par,
            f,
            instrument_strength=instrument_strength,
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

        if isinstance(method, IMPFunctionNonLin):
            method.causal_function = f

        if isinstance(method, BCF) or isinstance(method, BCFMLP):
            method.fit(np.hstack([X_train, Z_train]), y_train, seed=seed_torch)
        elif isinstance(method, ConstantFunc):
            method.fit(X_train, y_train, Z_train)
        else:
            method.fit(X_train, y_train, seed=seed_torch)

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
    config_path="../configs/exp-regularization",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    # Set seeds
    rng_numpy = np.random.default_rng(cfg.seed_numpy)

    ss = SeedSequence(cfg.seed_torch)  # master seed
    children = ss.spawn(cfg.n_reps)  # 10 child sequences
    seeds_torch = [int(c.generate_state(1)[0]) for c in children]  # 10 python ints

    #  Generate random function f and oracle quantities
    f = radial2D(num_basis=cfg.num_basis, seed=rng_numpy)
    X, _, Z, S, M, gamma = generate_data_radial_f(
        cfg.n_train,
        cfg.int_train,
        f,
        instrument_strength=cfg.instrument_strength,
        noise_sd=cfg.noise_sd,
        seed=rng_numpy,
    )
    Z = Z[:, np.newaxis]

    # Define factories for methods
    fx_factory = lambda x: MLP(in_dim=x, hidden=[64], activation=nn.Sigmoid)  # type: ignore
    fx_imp_factory = lambda x: MLP(in_dim=x, hidden=[64], activation=nn.Sigmoid)  # type: ignore
    gv_factory = lambda x: MLP(in_dim=x, hidden=[64], activation=nn.Sigmoid)  # type: ignore
    fx_linear_factory = lambda x: MLP(in_dim=x, hidden=[], activation=nn.Sigmoid)  # type: ignore

    # Define methods
    methods = [
        (
            "BCF-RF",
            BCF(
                n_exog=Z.shape[1],
                continuous_mask=np.repeat(True, X.shape[1]),
                passes=5,
                fx=XGBRegressor(learning_rate=0.025, base_score=0.0),
                gv=XGBRegressor(learning_rate=0.025, base_score=0.0),
                fx_imp=XGBRegressor(learning_rate=0.05, base_score=0.0),
            ),
        ),
        (
            "OLS-RF",
            OLS(fx=RandomForestRegressor()),
        ),
        (
            "IMP",
            IMPFunctionNonLin(
                causal_function=f,
                instrument_matrix=M,
                confounder_cov=S,
                confounder_effect=gamma,
            ),
        ),
        (
            "Causal",
            IMPFunctionNonLin(
                causal_function=f,
                instrument_matrix=M,
                confounder_cov=S,
                confounder_effect=gamma,
                use_imp=False,
            ),
        ),
        (
            "BCF-MLP",
            BCFMLP(
                n_exog=Z.shape[1],
                continuous_mask=np.repeat(True, X.shape[1]),
                fx_factory=fx_factory,
                fx_imp_factory=fx_imp_factory,
                gv_factory=gv_factory,
                epochs_step_1=1000,
                epochs_step_2=1500,
                lr_step_1=1e-3,
                lr_step_2=1e-4,
                weight_decay_step_1=2.5e-3,
                weight_decay_step_2=0.0,
            ),
        ),
        (
            "CF-MLP",
            BCFMLP(
                n_exog=Z.shape[1],
                continuous_mask=np.repeat(True, X.shape[1]),
                fx_factory=fx_factory,
                fx_imp_factory=fx_imp_factory,
                gv_factory=gv_factory,
                epochs_step_1=1000,
                lr_step_1=1e-3,
                weight_decay_step_1=2.5e-3,
                predict_imp=False,
            ),
        ),
        (
            "CF-MLP-2",
            BCFMLP(
                n_exog=Z.shape[1],
                continuous_mask=np.repeat(True, X.shape[1]),
                fx_factory=fx_factory,
                fx_imp_factory=fx_imp_factory,
                gv_factory=gv_factory,
                epochs_step_1=1000,
                lr_step_1=1e-3,
                weight_decay_step_1=2.5e-2,
                predict_imp=False,
            ),
        ),
        (
            "CF-MLP-3",
            BCFMLP(
                n_exog=Z.shape[1],
                continuous_mask=np.repeat(True, X.shape[1]),
                fx_factory=fx_factory,
                fx_imp_factory=fx_imp_factory,
                gv_factory=gv_factory,
                epochs_step_1=1000,
                lr_step_1=1e-3,
                weight_decay_step_1=2.5e-1,
                predict_imp=False,
            ),
        ),
        (
            "CF-MLP-4",
            BCFMLP(
                n_exog=Z.shape[1],
                continuous_mask=np.repeat(True, X.shape[1]),
                fx_factory=fx_factory,
                fx_imp_factory=fx_imp_factory,
                gv_factory=gv_factory,
                epochs_step_1=1000,
                lr_step_1=1e-3,
                weight_decay_step_1=2.5e-0,
                predict_imp=False,
            ),
        ),
        (
            "CF-MLP-5",
            BCFMLP(
                n_exog=Z.shape[1],
                continuous_mask=np.repeat(True, X.shape[1]),
                fx_factory=fx_factory,
                fx_imp_factory=fx_imp_factory,
                gv_factory=gv_factory,
                epochs_step_1=1000,
                lr_step_1=1e-3,
                weight_decay_step_1=2.5e1,
                predict_imp=False,
            ),
        ),
        (
            "CF-Linear",
            BCFMLP(
                n_exog=Z.shape[1],
                continuous_mask=np.repeat(True, X.shape[1]),
                fx_factory=fx_linear_factory,
                fx_imp_factory=fx_imp_factory,
                gv_factory=fx_linear_factory,
                epochs_step_1=1000,
                lr_step_1=1e-3,
                weight_decay_step_1=2.5e-3,
                predict_imp=False,
            ),
        ),
        (
            "OLS-MLP",
            OLSMLP(
                continuous_mask=np.repeat(True, X.shape[1]),
                fx_factory=fx_factory,
                epochs=1000,
                lr=1e-3,
                weight_decay=1e-3,
            ),
        ),
        (
            "Constant",
            ConstantFunc(),
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
            instrument_strength=cfg.instrument_strength,
            noise_sd=cfg.noise_sd,
            int_train=cfg.int_train,
            ints_test=cfg.ints_test,
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

# %%
# %%
# %%
# %%
# %%
