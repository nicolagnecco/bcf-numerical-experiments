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

TEST_RUN = True
LAST_TASK = 2
SEQUENTIAL = False
POOLSIZE = 3
DRYRUN = False

# Params
P = 3  # Number of predictors
C = 0  # Number of confounders
R = 1  # Number of training environments
NUM_SETS = 3  # Number of sets of training environments
ITERATIONS = 1  # Number of subsamples
N_OBS_SUBSAMPLED = 1000
SEED = 3430  # from https://www.random.org/integers
PRED_SELECTOR = partial(ds.select_top_predictors_lasso, environment_column="Z")
ENV_SELECTOR = partial(ds.random_env_selector, num_sets=NUM_SETS, seed=SEED)

# %%
# Paths
INPUT_DATA = "../data/processed/genes.csv"
INPUT_CONFIG = "configs/genes_extended_parallel_config.py"
RESULT_DIR = f"../results/try-extended/n_preds_{P}-n_conf_{C}-n_trainenv_{R}"
RESULT_NAME = "causalbench-res.csv"
OUTPUT_CONFIG = "_configs.py"


# Algorithms
base_model = RandomForestRegressor()
base_model_fx = LinearRegression()
base_model_gv = LinearRegression()
base_model_imp = RandomForestRegressor()
BCF_0 = BCF(
    n_exog=0,  # needs to know Z
    continuous_mask=np.repeat(True, 0),  # needs to know X
    fx=base_model_fx,
    gv=base_model_gv,
    fx_imp=base_model_imp,
    passes=2,
    alphas=np.array([0]),
)

BCF_1 = BCF(
    n_exog=0,  # needs to know Z
    continuous_mask=np.repeat(True, 0),  # needs to know X
    fx=base_model_fx,
    gv=base_model_gv,
    fx_imp=base_model_fx,
    passes=2,
    alphas=np.array([0]),
)

OLS_0 = OLS(fx=base_model)

OLS_1 = OLS(fx=base_model_fx)


algorithms = [
    (
        "BCF",
        ModelWrapper(BCF_0),
    ),
    (
        "BCF-lin",
        ModelWrapper(BCF_1),
    ),
    ("OLS", ModelWrapper(OLS_0)),
    ("OLS-lin", ModelWrapper(OLS_1)),
    ("ConstFunc", ModelWrapper(ConstantFunc())),
]


# Conds
if not (TEST_RUN):
    LAST_TASK = 10000000000
