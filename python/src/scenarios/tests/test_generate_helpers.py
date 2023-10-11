#%%
import numpy as np
from scipy.linalg import orth, null_space
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from src.scenarios.generate_helpers import decompose_mat, generate_orthonormal_vector
from src.scenarios.generate_x import generate_mu
from src.scenarios.generate_y import generate_gamma


#%%
nrep = 10000
res = np.zeros((nrep, 2))
thetas = np.zeros(nrep)


for i in range(nrep):
    theta = np.random.uniform(low=0, high=2 * np.pi)
    # x = np.random.uniform(low=-1, high=1, size=(2,))
    r = 1  # np.random.uniform(low=0, high=1)
    res[i, :] = np.array([r * np.cos(theta), r * np.sin(theta)])
    thetas[i] = theta
# %%
# plt.clf()
# plt.figure()
# plt.scatter(x=res[:, 0], y=res[:, 1], alpha=0.01)
# plt.show()

# plt.clf()
# plt.figure()
# plt.hist(x=thetas)
# # %%

# lam = np.linspace(0, 1, 4)
# # lam = lam / np.sqrt(lam ** 2 + (1 - lam)**2)

# plt.clf()
# plt.scatter(x=lam, y=lam)
# %%
