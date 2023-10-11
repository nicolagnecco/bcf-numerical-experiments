#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.scenarios.generate_data import generate_data, generate_data_Z_Gaussian
from src.scenarios.generate_y import generate_gamma
from src.scenarios.generate_helpers import decompose_mat, generate_orthonormal_vector

sns.set_theme()

# Constants
n = 1000
p = 5
p_effective = 3
tree_depth = 3
r = 2  # 50 obs per env
gamma_norm = 1.5
sd_y = 0.1
interv_strength = 100


# Generate data
(
    X_train,
    X_test,
    y_train,
    y_test,
    Z_train,
    f_train,
    f_test,
    *rest,
) = generate_data_Z_Gaussian(
    n=n,
    p=p,
    p_effective=p_effective,
    tree_depth=tree_depth,
    r=r,
    interv_strength=interv_strength,
    gamma_norm=gamma_norm,
    sd_y=sd_y,
)

# Create dataframes
x_cols = ["X{j}".format(j=j + 1) for j in range(p)]
col_names = x_cols + ["y", "f_X"]

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
        # np.repeat(r, n)[:, np.newaxis],
        f_test[:, np.newaxis],
    ]
)

dat_train = pd.DataFrame(data=mat_train, columns=col_names)
dat_train["env"] = "train"

dat_test = pd.DataFrame(data=mat_test, columns=col_names)
dat_test["env"] = "test"

dat = pd.concat([dat_test, dat_train])

#%%
# Plot some data
plt.clf()
g = sns.pairplot(
    data=dat.reset_index(drop=True).loc[:, x_cols + ["env"]],
    hue="env",
    kind="scatter",
    plot_kws=dict(s=80, edgecolor="white", linewidth=2.5, alpha=0.3),
)
# g.set(xlim=(-1, 1), ylim=(-1, 1))

#%%
plt.clf()
sns.scatterplot(data=dat, x="X1", y="y", hue="env", alpha=0.5)
sns.scatterplot(
    x=dat["X1"].values,
    y=dat["f_X"].values,
    label="causal",
    color="black",
    alpha=0.75,
    linestyle="dashed",
)


# %%
plt.clf()
sns.scatterplot(data=dat, x="X3", y="y", hue="Z")
sns.scatterplot(data=dat, x="X3", y="f_X")


#%%
plt.clf()
sns.scatterplot(data=dat, x="X1", y="X2", hue="env")


# %%
M = np.array([[1]])
Q, R = decompose_mat(M[:, np.arange(1)])

alpha = generate_orthonormal_vector(R.shape[1])
beta = generate_orthonormal_vector(Q.shape[1])


# %%
