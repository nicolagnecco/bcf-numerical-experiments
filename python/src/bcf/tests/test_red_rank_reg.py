# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import src.bcf.boosted_control_function_2 as bcf
import src.bcf.reduced_rank_regression as rrr
import src.stabletree.tree as st
from sklearn.model_selection import GridSearchCV
from src.scenarios.generate_data import generate_data, generate_data_Z_Gaussian
from src.scenarios.generate_helpers import decompose_mat

# Graphical settings
sns.set_theme(style="darkgrid")

# Constants
plot_var = 1
n = 1000
p = 4
p_effective = 1
tree_depth = 3
r = 2  # 50 obs per env
q = 1
gamma_norm = 2
sd_y = 0.1
interv_strength = 10
seed = None


# Function definitions
def eval_model(model, X_test, y_test):
    y_hat = model.predict(X_test)
    mse = float(((y_hat - y_test) ** 2).mean())
    return mse


#  Define methods
methods = [
    ("BCF", bcf.BCF()),
    ("OLS", bcf.OLS()),
]

# Generate data
(
    X_train,
    X_test,
    y_train,
    y_test,
    Z_train,
    f_train,
    f_test,
    *other,
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
)

Z_train_one_hot = Z_train

full_X = np.vstack((X_train, X_test))
full_f = np.concatenate((f_train, f_test))

# %%
X = X_train
Z = Z_train

redrank = rrr.RRR(alpha=0)

params = {"alpha": 10 ** np.arange(-5.0, 6)}

clf = GridSearchCV(redrank, params, scoring="neg_root_mean_squared_error")

# %%
clf.fit(Z, X)

print(clf.best_params_)


# %%

print(f"k_opt = {clf.best_estimator_.k_opt_}")
decompose_mat(clf.best_estimator_.M_hat_)

# %%
M_0 = other[1]
decompose_mat(M_0)  # type: ignore

# %%
M_ols = np.linalg.lstsq(a=Z, b=X, rcond=None)[0].T
decompose_mat(M_ols)


# %%
M_ols - clf.best_estimator_.M_hat_
