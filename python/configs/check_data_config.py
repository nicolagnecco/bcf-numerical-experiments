# Imports
from functools import partial

import numpy as np
import src.data.data_selectors as ds
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.algorithms.algorithm_wrapper import ModelWrapper
from src.algorithms.oracle_methods import ConstantFunc
from src.bcf.boosted_control_function_2 import BCF, OLS
from xgboost.sklearn import XGBRegressor

# Purpose: Check how data looks like
base_model = LinearRegression()  # RandomForestRegressor(n_estimators=20)


# Runs
TEST_RUN = False
FIRST_GENE = 22
LAST_GENE = 23

# Paths
INPUT_CONFIG = "configs/check_data_config.py"
RESULT_DIR = "../results/check_data/"
RESULT_NAME = "res.csv"
DATA_NAME = "data.csv"
OUTPUT_CONFIG = "configs.py"


# Data
n = 1000
p = 10
p_effective = 1
tree_depth = 1
r = 1
interv_strength = 10
gamma_norm = 1.5
sd_y = 0.0

n_sims = 10

SEED = 1057  # from https://www.random.org/integers


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
