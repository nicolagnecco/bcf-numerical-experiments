# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from src.algorithms.anchor_regression import AnchorRegression, AnchorRegressionCV
from ucimlrepo import fetch_ucirepo, list_available_datasets

# %%
dat = fetch_ucirepo(name="Bike Sharing")
# %%
data = dat["data"]["original"]
# %%
y = (np.sqrt(data["cnt"])).to_numpy()
X = data[
    [
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "holiday",
        "weekday",
    ]
]
Z = data[["dteday"]]
# %%
categorical_features = ["holiday", "weekday"]
preprocessor_X = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(drop="first", sparse=False), categorical_features)
    ],
    remainder="passthrough",
)

preprocessor_Z = OneHotEncoder(drop="first", sparse=False)

# %%
X_ = preprocessor_X.fit_transform(X)
Z_ = preprocessor_Z.fit_transform(Z)
cont_mask = np.concatenate((np.repeat(False, 7), np.repeat(True, 4)))
# %%
anchor = AnchorRegression(n_exog=Z_.shape[1], continuous_mask=cont_mask, gamma=1.0)

# %%
anchor.fit(np.hstack((X_, Z_)), y)  # type: ignore
# %%
y_hat = anchor.predict(X_)  # type: ignore
# %%
y_hat[0]

# %%
linear_reg = LinearRegression()

linear_reg.fit(X_, y)
# %%
y_hat_lr = linear_reg.predict(X_)
# %%
y_hat_lr[0]
# %%
anchor_cv = AnchorRegressionCV(
    n_exog=Z_.shape[1], continuous_mask=cont_mask, gammas=[1.0, 10.0, 100.0]
)
# %%
anchor_cv.fit(np.hstack((X_, Z_)), y)  # type:ignore
# %%
y_hat_cv = anchor_cv.predict(X_)  # type:ignore
# %%
import matplotlib.pyplot as plt

plt.scatter(y_hat_cv, y_hat, alpha=0.5)
# plt.scatter(y_hat, y, color='red')
# %%
from anchorboosting import AnchorBooster

anchor_booster = AnchorBooster(gamma=1.0)
# %%
anchor_booster.fit(X, y, Z_, categorical_feature=["holiday", "weekday"])
# %%
