# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ortho_group, zscore
from src.scenarios.generate_helpers import decompose_mat
from src.scenarios.generate_x import (
    generate_mu,
    generate_random_covariance,
    generate_random_mat,
    sample_X,
)

p = 1
n = 1000
rank = 0
nrep = 10

M = generate_random_mat(5, 4, q=1, eigengap=8.5)

# %%
xvars = np.zeros(shape=(nrep, p))
for i in range(nrep):
    X = sample_X(n=n, p=p, r=rank)
    print(X.var(axis=0).sum())

# %%
mu = generate_mu(n=n, p=p, r=rank)
V = sample_V(n=n, p=p)

X = mu + V

colsM = ["mu{j}".format(j=j + 1) for j in range(p)]
envs = pd.DataFrame(data=mu, columns=colsM)
env_names = envs.drop_duplicates().copy()
env_names["index"] = env_names.index

colsX = ["X{j}".format(j=j + 1) for j in range(p)]
dat = (
    pd.DataFrame(data=X, columns=colsX)
    .join(envs)
    .merge(env_names, on=colsM, how="left")
)


# plt.clf()
# if p == 1:
#     sns.histplot(data=dat, x="X1", hue="index")
# else:
#     sns.scatterplot(data=dat, x="X1", y="X2", hue="index")

# sns.scatterplot(data=dat, x="mu1", y="mu2", hue="index")

# %%
import numpy as np
from scipy.stats import ortho_group
from sklearn.preprocessing import StandardScaler

p = 4
r = 3
Z_sample_space = np.hstack([np.zeros((p, 1)), np.eye(p)])
Z = Z_sample_space[:, np.random.choice(a=r + 1, size=n)]

if p == 1:
    M = np.array([[1]])
else:
    M = ortho_group.rvs(p)

mu = (M @ Z).T

mu = StandardScaler(with_std=False).fit_transform(X=mu)
mu2 = (M @ StandardScaler(with_std=False).fit_transform(X=Z.T).T).T
# %%


# %%
