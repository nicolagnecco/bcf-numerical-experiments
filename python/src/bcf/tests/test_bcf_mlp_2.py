# %%
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from numpy.random import BitGenerator, Generator, SeedSequence
from sklearn.ensemble import RandomForestRegressor
from src.algorithms.oracle_methods import IMPFunctionNonLin
from src.bcf.boosted_control_function_2 import BCF, OLS
from src.bcf.boosted_control_function_mlp import BCFMLP, OLSMLP
from src.bcf.mlp import MLP
from src.scenarios.generate_data import generate_data_radial_f
from src.scenarios.generate_helpers import radial2D

# Graphical settings
sns.set_theme(style="darkgrid")


# %% ---- Function definitions
def eval_model(y_hat, y_test):
    mse = float(((y_hat - y_test) ** 2).mean())
    return mse


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


def plot_data(X, y, f):
    # Define grid
    x1 = np.linspace(-10, 10, 100)
    x2 = np.linspace(-10, 10, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])

    # Evaluate f on the grid
    Z_surf = f(X_grid).reshape(X1.shape)

    # --- build interactive figure
    fig = go.Figure()

    # surface for f(x)
    fig.add_trace(
        go.Surface(x=X1, y=X2, z=Z_surf, opacity=0.7, showscale=True, name="f(x)")
    )

    # scatter points for generated data
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=y,
            mode="markers",
            marker=dict(size=3, color="red", opacity=0.15),
            name="samples",
        )
    )

    fig.update_layout(
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="y"),
        title="Interactive 3D view of f(x) and generated data",
    )

    fig.show()


# %% ---- Constant definitions
use_plotly = False
p = 2
n_train = 1000
n_test = 1000
intvec = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 3.99]
num_basis = 10
rng = np.random.default_rng(None)
seed_torch = 42


# %% Generate random function f and oracle quantities
f = radial2D(num_basis=num_basis, seed=rng)
X, y, Z, S, M, gamma = generate_data_radial_f(n_train, 3.5, f, noise_sd=0.1, seed=rng)
Z = Z[:, np.newaxis]
# %%
if use_plotly:
    plot_data(X, y, f)


# %% ---- define methods
fx_factory = lambda x: MLP(in_dim=x, hidden=[64], activation=nn.Sigmoid)  # type: ignore
fx_imp_factory = lambda x: MLP(in_dim=x, hidden=[64], activation=nn.Sigmoid)  # type: ignore
gv_factory = lambda x: MLP(in_dim=x, hidden=[64], activation=nn.Sigmoid)  # type: ignore

#  Define methods
methods = [
    (
        "BCF",
        BCF(
            n_exog=Z.shape[1],
            continuous_mask=np.repeat(True, X.shape[1]),
            passes=5,
            fx=RandomForestRegressor(),
            fx_imp=RandomForestRegressor(),
        ),
    ),
    (
        "OLS",
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
            weight_decay_step_1=1e-3,
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
            weight_decay_step_1=0e-3,
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
            weight_decay_step_1=2.5e-1,
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
]


# %%  generate datasets
X_train, y_train, Z_train, _, _, _ = generate_data_radial_f(
    n_train, 0.5, f, noise_sd=0.1, seed=rng
)
Z_train = Z_train[:, np.newaxis]

# %%
test_datasets = [
    generate_data_radial_f(n_train, int_par, f, noise_sd=0.1, seed=rng)
    for int_par in intvec
]

mses = pd.DataFrame(columns=["model", "test-mse", "int_par"])
col_names = ["X{j}".format(j=j + 1) for j in range(p)] + ["y", "y_hat"]
dat_methods = pd.DataFrame(columns=col_names + ["model", "env", "int_par"])

# %% fit/predict/evaluate methods
for method_name, method in methods[4:6]:
    # fit methods
    print(method_name)
    if method_name in ["BCF", "BCF-MLP", "CF-MLP", "CF-MLP-2"]:
        method.fit(np.hstack([X_train, Z_train]), y_train, seed=seed_torch)
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
    for i, int_par in enumerate(intvec):
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

        test_mse = eval_model(y_hat, y)

        mses = pd.concat(
            [
                mses,
                pd.DataFrame.from_dict(
                    {
                        "model": [method_name],
                        "test-mse": [test_mse],
                        "int_par": [int_par],
                    }
                ),
            ],
            ignore_index=True,
        )
        # print(
        #     "Test MSE of {method_name}: {test_mse:.2f}".format(
        #         method_name=method_name, test_mse=test_mse, int_par=int_par
        #     )
        # )

# %%
# Plot data
plot_var = 1
js = [0.5, 1.5, 3.5]

dat2plot = dat_methods[dat_methods["env"] == "test"]
# %%
plt.figure()

sns.scatterplot(
    data=dat2plot[(dat2plot["model"] == "CF-MLP") & (dat2plot["int_par"].isin(js))],
    x=f"X{plot_var}",
    # hue="int_par",
    y="y",
    alpha=0.5,
    hue="int_par",
    color="black",
)

filter_series = dat2plot["model"].isin(["CF-MLP", "CF-MLP-2"]) & (
    dat2plot["int_par"].isin(js)
)
sns.scatterplot(
    data=dat2plot[filter_series],
    x=f"X{plot_var}",
    # hue="int_par",
    y="y_hat",
    hue="model",
    alpha=0.5,
)

# %%
plt.figure()

sns.lineplot(
    data=mses,
    x="int_par",
    y="test-mse",
    hue="model",
    estimator=None,
)
# %%
# %%
# %%
