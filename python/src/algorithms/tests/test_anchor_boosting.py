# %%
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from anchorboosting import AnchorBooster
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from src.bcf.boosted_control_function_2 import BCF, OLS
from src.scenarios.generate_data import generate_data_radial_f
from src.scenarios.generate_helpers import radial2D
from xgboost import XGBRegressor

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
X, y, Z, S, M, gamma = generate_data_radial_f(
    n_train, 3.5, f, instrument_strength=1.0, noise_sd=0.1, seed=rng
)
Z = Z[:, np.newaxis]
# %%
if use_plotly:
    plot_data(X, y, f)


# %% ---- define methods
fx_factory = lambda x: MLP(in_dim=x, hidden=[64], activation=nn.Sigmoid)  # type:ignore

#  Define methods
methods = [
    (
        "BCF-new",
        BCF(
            n_exog=1,
            continuous_mask=np.repeat(True, X.shape[1]),
            fx=XGBRegressor(learning_rate=0.05, base_score=0.0),
            gv=XGBRegressor(learning_rate=0.01, base_score=0.0),
            fx_imp=XGBRegressor(learning_rate=0.05, base_score=0.0),
            passes=10,
        ),
    ),
    # (
    #     "GroupDRO",
    #     GroupDRO(fx_factory=fx_factory, n_groups=4, n_epochs=500),
    # ),
    (
        "OLS",
        OLS(fx=RandomForestRegressor()),
    ),
    (
        "AnchorBooster",
        AnchorBooster(
            gamma=200, num_boost_round=1000, max_depth=3, min_gain_to_split=0.1
        ),
    ),
]


# %%  generate datasets
X_train, y_train, Z_train, _, _, _ = generate_data_radial_f(
    n_train, 0.5, f, instrument_strength=1.0, noise_sd=0.1, seed=rng
)
Z_train = Z_train[:, np.newaxis]

# %%
test_datasets = [
    generate_data_radial_f(
        n_train, int_par, f, instrument_strength=1.0, noise_sd=0.1, seed=rng
    )
    for int_par in intvec
]

mses = pd.DataFrame(columns=["model", "test-mse", "int_par"])
col_names = ["X{j}".format(j=j + 1) for j in range(p)] + ["y", "y_hat"]
dat_methods = pd.DataFrame(columns=col_names + ["model", "env", "int_par"])


# %% fit/predict/evaluate methods
for method_name, method in methods:
    # fit methods
    print(method_name)
    if isinstance(method, AnchorBooster):
        method.fit(X_train, y_train, Z_train)
    elif isinstance(method, BCF):
        method.fit(np.hstack([X_train, Z_train]), y_train)
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
js = [0.5, 3.5]

dat2plot = dat_methods[dat_methods["env"] == "test"]
# %%
plt.figure()

sns.scatterplot(
    data=dat2plot[(dat2plot["model"] == "OLS") & (dat2plot["int_par"].isin(js))],
    x=f"X{plot_var}",
    # hue="int_par",
    y="y",
    alpha=0.5,
    hue="int_par",
    color="black",
)

filter_series = dat2plot["model"].isin(["OLS", "BCF-new", "AnchorBooster"]) & (
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
