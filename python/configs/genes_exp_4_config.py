# Imports
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

# Purpose: Try lin reg with 5 predictors + up to 1 envs + push quantile higher
base_model = (
    LinearRegression()
)  # GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1)

# Runs
TEST_RUN = False
FIRST_GENE = 6
LAST_GENE = 7

# Paths
INPUT_DATA = "../data/processed/genes.csv"
INPUT_CONFIG = "configs/genes_exp_4_config.py"
RESULT_DIR = "../results/output_data/"
RESULT_NAME = "causalbench-res.csv"
DATA_NAME = "causalbench-data.csv"
OUTPUT_CONFIG = "configs.py"


# Data
QUANTILE_THRES = 0.025
N_OBS_SUBSAMPLED = 1000
N_TOP_PREDS = 5
N_TOP_ENVS = 5
START_ENV = 1
PRED_SELECTOR = partial(
    ds.select_top_predictors, n_top_pred=N_TOP_PREDS, environment_column="Z"
)
SEED = 235142  # from https://www.random.org/integers


# Algorithms
BCF_0 = BCF(
    n_exog=0,  # needs to know Z
    continuous_mask=np.repeat(True, 0),  # needs to know X
    fx=base_model,
    gv=base_model,
    fx_imp=base_model,
    passes=2,
)

OLS_0 = OLS(fx=base_model)

algorithms = [
    (
        "BCF",
        ModelWrapper(BCF_0),
    ),
    ("OLS", ModelWrapper(OLS_0)),
    ("ConstFunc", ModelWrapper(ConstantFunc())),
]


# Conds
if TEST_RUN:
    first_gene = FIRST_GENE
    last_gene = LAST_GENE
else:
    first_gene = 0
    last_gene = 10000
