# %%
import pandas as pd
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
# %%
X["MedHouseVal"] = y


# %%
X.to_csv(path_or_buf="../data/processed/housing-sklearn.csv", index=False)
