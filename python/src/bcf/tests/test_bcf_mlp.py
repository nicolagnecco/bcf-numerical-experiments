# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from src.algorithms.oracle_methods import CausalFunction, ConstantFunc, IMPFunction
from src.bcf.boosted_control_function_2 import BCF, OLS
from src.bcf.boosted_control_function_mlp import BCFMLP
from src.bcf.mlp import MLP
from src.scenarios.generate_data import generate_data, generate_data_Z_Gaussian
from src.scenarios.generate_helpers import decompose_mat

# Graphical settings
sns.set_theme(style="darkgrid")

# Constants
plot_var = 2
n = 1000
p = 4
p_effective = 2
tree_depth = 3
r = 3  # 50 obs per env
q = 1
gamma_norm = 2
sd_y = 0.1
interv_strength = 10
seed = None


# Function definitions
def eval_model(y_hat, y_test):
    mse = float(((y_hat - y_test) ** 2).mean())
    return mse


def create_dataframe(
    X: np.ndarray, y: np.ndarray, env: str, model: str
) -> pd.DataFrame:
    dat = pd.DataFrame(
        data=np.concatenate([X, y[:, np.newaxis]], axis=1), columns=col_names
    )
    dat["model"] = model
    dat["env"] = env
    return dat


# %%
# Generate data
(
    X_train,
    X_test,
    y_train,
    y_test,
    Z_train,
    Z_test,
    f_train,
    f_test,
    f_tree,
    M,
    S,
    gamma,
) = generate_data_Z_Gaussian(
    n=n,
    p=p,
    q=q,
    p_effective=p_effective,
    tree_depth=tree_depth,
    r=r,
    interv_strength=interv_strength,
    gamma_norm=gamma_norm,
    sd_y=sd_y,
    seed=seed,
    mean_X=0,
    mean_Z=0,
)
# %%
Z_train_one_hot = Z_train

full_X = np.vstack((X_train, X_test))
full_f = np.concatenate((f_train, f_test))
full_Z = np.vstack((Z_train, Z_test))


# %%
fx_factory = lambda x: MLP(in_dim=x, hidden=[128, 64], activation=nn.Sigmoid)  # type: ignore
fx_imp_factory = lambda x: MLP(in_dim=x, hidden=[128, 64], activation=nn.Sigmoid)  # type: ignore
gv_factory = lambda x: MLP(in_dim=x, hidden=[128, 64], activation=nn.Sigmoid)  # type: ignore

# %%
#  Define methods
methods = [
    (
        "BCF-new",
        BCF(
            n_exog=Z_train_one_hot.shape[1],
            continuous_mask=np.repeat(True, X_train.shape[1]),
            passes=2,
            fx=RandomForestRegressor(),
            fx_imp=RandomForestRegressor(),
        ),
    ),
    (
        "OLS",
        OLS(fx=RandomForestRegressor()),
    ),
    # ("ControlTwicing", st.ControlTwicing(is_imp=True)),
    (
        "IMP",
        IMPFunction(
            causal_function=f_tree,
            instrument_matrix=M,
            confounder_cov=S,
            confounder_effect=gamma,
        ),
    ),
    (
        "BCF_MLP",
        BCFMLP(
            n_exog=Z_train_one_hot.shape[1],
            continuous_mask=np.repeat(True, X_train.shape[1]),
            fx_factory=fx_factory,
            fx_imp_factory=fx_imp_factory,
            gv_factory=gv_factory,
            epochs_step_1=500,
            epochs_step_2=1000,
            lr_step_1=1e-3,
            lr_step_2=1e-4,
            weight_decay_step_1=1e-4,
            weight_decay_step_2=0.0,
        ),
    ),
    (
        "CF_MLP",
        BCFMLP(
            n_exog=Z_train_one_hot.shape[1],
            continuous_mask=np.repeat(True, X_train.shape[1]),
            fx_factory=fx_factory,
            fx_imp_factory=fx_imp_factory,
            gv_factory=gv_factory,
            epochs_step_1=500,
            lr_step_1=1e-3,
            weight_decay_step_1=1e-4,
            predict_imp=False,
        ),
    ),
]


# %%
# Fit - predict methods
col_names = ["X{j}".format(j=j + 1) for j in range(p)] + ["y"]
dat_methods = pd.DataFrame(columns=col_names + ["model", "env"])


# %%

mses = pd.DataFrame(columns=["model", "test-mse"])

for method_name, method in methods:
    # fit/predict
    print(method_name)
    if method_name == "OLS":
        method.fit(X_train, y_train)
    elif method_name == "IMP":
        method.fit(X_train, y_train, Z_train_one_hot)
    else:
        method.fit(np.hstack([X_train, Z_train_one_hot]), y_train)

    # ---- predict on training data and save
    y_hat = method.predict(X_train)

    inner_dat = create_dataframe(X=X_train, y=y_hat, env="train", model=method_name)

    dat_methods = pd.concat([dat_methods, inner_dat])

    # ---- predict on test data and save
    y_hat = method.predict(X_test)

    inner_dat = create_dataframe(X=X_test, y=y_hat, env="test", model=method_name)

    dat_methods = pd.concat([dat_methods, inner_dat])

    test_mse = eval_model(y_hat, y_test)

    mses = pd.concat(
        [mses, pd.DataFrame.from_dict({"model": [method_name], "test-mse": [test_mse]})]
    )
    print(
        "Test MSE of {method_name}: {test_mse:.2f}".format(
            method_name=method_name, test_mse=test_mse
        )
    )

# %%
# Plot data
col_names = ["X{j}".format(j=j + 1) for j in range(p)] + [
    "y",
    "f_X",
]
mat_train = np.hstack(
    [
        X_train,
        y_train[:, np.newaxis],
        f_train[:, np.newaxis],
    ]
)

mat_test = np.hstack(
    [
        X_test,
        y_test[:, np.newaxis],
        f_test[:, np.newaxis],
    ]
)

dat_train = pd.DataFrame(data=mat_train, columns=col_names)
dat_train["env"] = "train"

dat_test = pd.DataFrame(data=mat_test, columns=col_names)
dat_test["env"] = "test"

dat = pd.concat([dat_train, dat_test])

plt.figure()
sns.scatterplot(
    data=dat[dat["env"] == "train"],
    x=f"X{plot_var}",
    y="y",
    alpha=0.25,
    # hue="env",
    color="black",
    legend=False,  # type: ignore
)

filter_series = dat_methods["model"].isin(["IMP", "BCF_MLP"]) & (
    dat_methods["env"] == "test"
)
sns.scatterplot(
    data=dat_methods[filter_series],
    x=f"X{plot_var}",
    y="y",
    hue="model",
    alpha=0.5,
)
# sns.lineplot(
#     x=full_X[:, plot_var - 1],
#     y=full_f,
#     label="causal",
#     color="black",
#     alpha=0.75,
#     linestyle="dashed",
# )

# %%
print(f"True q: {q}")
print(f"q_hat: {(methods[0])[1].q_opt_}")

# %%
# %%
# %%
# %%
# %%
# %%
# %%
