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

# Purpose: Try to see what happens for increasing intervention strength
base_model = RandomForestRegressor()
base_model_fx = LinearRegression()
base_model_gv = LinearRegression()
base_model_imp = RandomForestRegressor()
# GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1)

# Runs
TEST_RUN = False
FIRST_GENE = 0
LAST_GENE = 3

# Data
ADD_CONFOUNDER = False
QUANTILE_THRES = 0.025
N_OBS_SUBSAMPLED = 1000
N_TOP_PREDS = 3
N_ENVS = 1
PRED_SELECTOR = partial(ds.select_top_predictors_lasso, environment_column="Z")
ENV_SELECTOR = ds.select_environment_e
CONF_SELECTOR = ds.select_confounder_as_target_predictor
SEED = 3430  # from https://www.random.org/integers


# Paths
INPUT_DATA = "../data/processed/genes.csv"
INPUT_CONFIG = "configs/genes_exp_parallel_config.py"
RESULT_DIR = (
    f"../results/discuss-genes/confounders_{ADD_CONFOUNDER}-npreds_{N_TOP_PREDS}"
)
RESULT_NAME = "causalbench-res.csv"
DATA_NAME = "causalbench-data.csv"
OUTPUT_CONFIG = "configs.py"


# Algorithms
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
if TEST_RUN:
    first_gene = FIRST_GENE
    last_gene = LAST_GENE
else:
    first_gene = 0
    last_gene = 10000
