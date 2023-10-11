# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ortho_group
from sklearn.datasets import make_friedman1, make_friedman2
from src.scenarios.generate_helpers import decompose_mat, generate_orthonormal_vector
from src.scenarios.generate_x import generate_M, generate_mu, sample_V
from src.scenarios.generate_y import generate_gamma, generate_tree

n = 100
p = 3
r = 2
t = 0.2
seed = 42
gamma_norm = 1


def test_gamma():
    M = generate_M(p, seed)

    np.random.seed(1991)
    Q, R = decompose_mat(M[:, np.arange(r)])
    alpha = generate_orthonormal_vector(p - r)
    beta = generate_orthonormal_vector(r)

    gamma = t * R @ alpha + (1 - t) * Q @ beta
    gamma = gamma / np.sqrt(np.sum(gamma**2))

    np.random.seed(1991)
    assert np.all(generate_gamma(M, r, t, gamma_norm) == gamma)


# %%
np.random.seed(19234)
depth = 3
p = 10
p_effective = 2
r = 1
n = 10000
noise = 0.02
conf_strength = 0.2
k = 1
mytree = generate_tree(p_effective, depth)
mu = generate_mu(n=n, p=p, r=r)
V = sample_V(n=n, p=p)
X = k * mu + V
gamma = np.ones(p) / np.sqrt(p)

f_X = mytree.predict(X=X)
y = f_X + V @ gamma + noise * np.random.normal(size=n)
np.var(V @ gamma)

colsM = ["mu{j}".format(j=j + 1) for j in range(p)]
envs = pd.DataFrame(data=mu, columns=colsM)
env_names = envs.drop_duplicates().copy()
env_names["index"] = env_names.index

colsX = ["X{j}".format(j=j + 1) for j in range(p)]
dat = (
    pd.DataFrame(data=X, columns=colsX)
    .join(envs)
    .merge(env_names, on=colsM, how="left")
    .join(pd.DataFrame(data=y, columns=["y"]))
    .join(pd.DataFrame(data=f_X, columns=["f_X"]))
)


# plt.clf()
# if p == 1:
#     sns.histplot(data=dat, x="X1", hue="index")
# else:
#     sns.scatterplot(data=dat, x="X1", y="X2", hue="index")

# #%%
# plt.clf()
# sns.scatterplot(data=dat, x="X1", y="y", hue="index")
# sns.scatterplot(data=dat, x="X1", y="f_X")
# %% other stuff
(X, y) = make_friedman1(noise=0)

# %%
n, p = X.shape
colnames = ["X_{j}".format(j=j + 1) for j in range(0, p)]
dat = pd.DataFrame(data=np.column_stack([X, y]), columns=colnames + ["y"])
# %%

# plt.clf()
# sns.scatterplot(data=dat, x="X_6", y="y")
# %%
