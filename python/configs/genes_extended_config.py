# %%Imports
from functools import partial

import numpy as np
import src.data.data_selectors as ds
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from src.algorithms.algorithm_wrapper import ModelWrapper
from src.algorithms.oracle_methods import ConstantFunc
from src.bcf.boosted_control_function_2 import BCF, OLS
from xgboost.sklearn import XGBRegressor

# Purpose: ...

TEST_RUN = False
LAST_TASK = 2
SEQUENTIAL = False

# Params
P = 1  # Number of predictors
C = 0  # Number of confounders
R = 1  # Number of training environments
NUM_SETS = 3  # Number of sets of training environments
ITERATIONS = 1  # Number of subsamples
N_OBS_SUBSAMPLED = 1000
SEED = 9234  # from https://www.random.org/integers
PRED_SELECTOR = partial(ds.select_top_predictors_lasso, environment_column="Z")
ENV_SELECTOR = partial(ds.random_env_selector, num_sets=NUM_SETS, seed=SEED)

# %%
# Paths
INPUT_DATA = "../data/processed/genes.csv"
INPUT_CONFIG = "configs/genes_extended_config.py"
RESULT_DIR = f"../results/try-extended/n_preds_{P}-n_conf_{C}-n_trainenv_{R}"
RESULT_NAME = "causalbench-res.csv"
OUTPUT_CONFIG = "_configs.py"


# Algorithms
def create_bcf_0():
    return ModelWrapper(
        BCF(
            n_exog=0,  # needs to know Z
            continuous_mask=np.repeat(True, 0),  # needs to know X
            fx=RandomForestRegressor(),
            gv=LinearRegression(),
            fx_imp=RandomForestRegressor(),
            passes=2,
            alphas=np.array([0]),
        )
    )


def create_bcf_1():
    return ModelWrapper(
        BCF(
            n_exog=0,  # needs to know Z
            continuous_mask=np.repeat(True, 0),  # needs to know X
            fx=LinearRegression(),
            gv=LinearRegression(),
            fx_imp=LinearRegression(),
            passes=2,
            alphas=np.array([0]),
        )
    )


def create_ols_0():
    return ModelWrapper(OLS(fx=RandomForestRegressor()))


def create_ols_1():
    return ModelWrapper(OLS(fx=LinearRegression()))


def create_const():
    return ModelWrapper(ConstantFunc())


algorithms = [
    ("BCF", create_bcf_0),
    ("BCF-lin", create_bcf_1),
    ("OLS", create_ols_0),
    ("OLS-lin", create_ols_1),
    ("ConstFunc", create_const),
]


# Conds
if not (TEST_RUN):
    LAST_TASK = 10000000000
