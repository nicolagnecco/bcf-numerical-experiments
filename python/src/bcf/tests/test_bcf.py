# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import src.bcf.boosted_control_function_2 as bcf2
import src.bcf.reduced_rank_regression as rrr
import src.stabletree.tree as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from src.scenarios.generate_data import generate_data, generate_data_Z_Gaussian
from src.scenarios.generate_helpers import decompose_mat

# Graphical settings
sns.set_theme(style="darkgrid")

# Constants
plot_var = 1
n = 1000
p = 10
p_effective = 1
tree_depth = 3
r = 3  # 50 obs per env
q = 2
gamma_norm = 2
sd_y = 0.1
interv_strength = 50
seed = None
Z_avail_test = False


# Function definitions
def eval_model(y_hat, y_test):
    mse = float(((y_hat - y_test) ** 2).mean())
    return mse


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

Z_train_one_hot = Z_train

full_X = np.vstack((X_train, X_test))
full_f = np.concatenate((f_train, f_test))
full_Z = np.vstack((Z_train, Z_test))

# %%
#  Define methods
methods = [
    (
        "BCF",
        bcf.BCF(
            passes=2,
            fx_imp_=RandomForestRegressor(),
            fx_=RandomForestRegressor(),
            Z_available_test=Z_avail_test,
        ),
    ),
    (
        "BCF-new",
        bcf2.BCF(
            n_exog=Z_train_one_hot.shape[1],
            continuous_mask=np.repeat(True, X_train.shape[1]),
            passes=2,
            fx=RandomForestRegressor(),
            fx_imp=RandomForestRegressor(),
        ),
    ),
    (
        "OLS",
        bcf.OLS(passes=2, Z_available_test=Z_avail_test),
    ),
    # ("ControlTwicing", st.ControlTwicing(is_imp=True)),
    (
        "IMP",
        st.IMPFunction(
            causal_function=f_tree,
            instrument_matrix=M,
            confounder_cov=S,
            confounder_effect=gamma,
        ),
    ),
]

# %%
# Fit - predict methods
dat_methods = pd.DataFrame(columns=["X", "y_hat", "model"])

for method_name, method in methods:
    # fit/predict
    if method_name == "BCF-new":
        method.fit(np.hstack([X_train, Z_train_one_hot]), y_train)
    else:
        method.fit(X_train, y_train, Z_train_one_hot)

    if Z_avail_test:
        Z_test_curr = Z_test
        Z_full_curr = full_Z
    else:
        Z_test_curr = None
        Z_full_curr = None

    if method_name == "IMP" or method_name == "BCF-new":
        y_hat = method.predict(full_X)
    else:
        y_hat = method.predict(full_X, Z_full_curr)

    inner_dat = pd.DataFrame.from_dict(
        {"X": full_X[:, plot_var - 1], "y_hat": y_hat, "model": method_name}
    )

    dat_methods = pd.concat([dat_methods, inner_dat])

    if method_name == "IMP" or method_name == "BCF-new":
        y_hat = method.predict(X_test)
    else:
        y_hat = method.predict(X_test, Z_test_curr)

    test_mse = eval_model(y_hat, y_test)
    print(
        "Test MSE of {method_name}: {test_mse:.2f}".format(
            method_name=method_name, test_mse=test_mse
        )
    )

# %%
# Plot data
col_names = ["X{j}".format(j=j + 1) for j in range(p)] + [
    "y",
    # "Z",
    "f_X",
]

mat_train = np.hstack(
    [
        X_train,
        y_train[:, np.newaxis],
        # Z_train[:, np.newaxis],
        f_train[:, np.newaxis],
    ]
)

mat_test = np.hstack(
    [
        X_test,
        y_test[:, np.newaxis],
        # np.repeat(n_envs, n)[:, np.newaxis],
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
    data=dat,
    x=f"X{plot_var}",
    y="y",
    alpha=0.25,
    # hue="env",
    color="grey",
    legend=False,  # type: ignore
)
sns.scatterplot(
    data=dat_methods[dat_methods["model"].isin(("BCF", "BCF-new"))],
    x="X",
    y="y_hat",
    hue="model",
    alpha=0.5,
)
sns.lineplot(
    x=full_X[:, plot_var - 1],
    y=full_f,
    label="causal",
    color="black",
    alpha=0.75,
    linestyle="dashed",
)

# %%
print(f"True q: {q}")
print(f"q_hat: {(methods[0])[1].k_opt_}")
# print(f"q_hat old: {(methods[2])[1].k_opt_}")

# %%
