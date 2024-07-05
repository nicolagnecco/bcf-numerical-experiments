# %%
import copy
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from src.algorithms.oracle_methods import CausalFunction, ConstantFunc, IMPFunction
from src.bcf.boosted_control_function_2 import BCF, OLS
from src.bcf.reduced_rank_regression import RRR
from src.scenarios.generate_data import generate_data_Z_Gaussian
from src.scenarios.generate_helpers import distance_left_nullspace

# %%
# Constant definitions
logging.basicConfig(filename="sims.log", level=logging.INFO)


# Function definition
def run_simulation(
    params: dict,
    methods: List[
        Tuple[
            str,
            Union[
                ConstantFunc,
                CausalFunction,
                IMPFunction,
                BCF,
                OLS,
            ],
        ]
    ],
) -> dict:
    # remove parameters not needed for `generate_data()`
    params = removekey("n_reps", params)

    params_generate_data = dict(
        filter(lambda elem: not elem[0].startswith("_"), params.items())
    )

    # generate data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        Z_train,
        Z_test,
        f_train,
        f_test,
        f_tree,
        M,
        S,
        gamma,
    ) = generate_data_Z_Gaussian(**params_generate_data)

    # fit-predict-evaluate methods
    mses = []
    method_names = []

    for method_name, method in methods:
        # fit methods
        if isinstance(method, CausalFunction):
            method = CausalFunction(causal_function=f_tree)
            method.fit(X=X_train, y=y_train, Z=Z_train)

        elif isinstance(method, IMPFunction):
            method = IMPFunction(
                causal_function=f_tree,
                instrument_matrix=M,
                confounder_cov=S,
                confounder_effect=gamma,
            )
            method.fit(X=X_train, y=y_train, Z=Z_train)

        elif isinstance(method, ConstantFunc):
            method.fit(X=X_train, y=y_train, Z=Z_train)

        elif isinstance(method, BCF):
            method.fit(X=np.hstack([X_train, Z_train]), y=y_train)

        elif isinstance(method, OLS):
            method.fit(X=X_train, y=y_train)

        # predict and evaluate
        y_hat = method.predict(X_test)
        mses.append(compute_mse(y_test, y_hat))
        method_names.append(method_name)

    # return dictionary with simulation results
    return {
        "method_names": method_names,
        "MSE": mses,
    }


def compute_mse(y, y_hat) -> float:
    return float(((y - y_hat) ** 2).mean())


def removekey(key: str, dict: dict) -> dict:
    new_dict = copy.deepcopy(dict)
    del new_dict[key]

    return new_dict


def run_simulation_rank(
    params: dict,
    methods: List[Tuple[str, RRR, dict, str],],
) -> dict:
    # remove parameters not needed for `generate_data()`
    params = removekey("n_reps", params)

    params_generate_data = dict(
        filter(lambda elem: not elem[0].startswith("_"), params.items())
    )

    # generate data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        Z_train,
        Z_test,
        f_train,
        f_test,
        f_tree,
        M,
        S,
        gamma,
    ) = generate_data_Z_Gaussian(**params_generate_data)

    # fit-predict-evaluate methods
    k_opts = []
    method_names = []
    dist_left_null = []

    for method_name, method, param_grid, scoring in methods:
        # fit/predict
        clf = GridSearchCV(estimator=method, param_grid=param_grid, scoring=scoring)
        clf.fit(Z_train, X_train)
        k_opts.append(clf.best_estimator_.k_opt_)
        dist_left_null.append(distance_left_nullspace(M, clf.best_estimator_.M_hat_))
        method_names.append(method_name)

    # return dictionary with simulation results
    return {
        "method_names": method_names,
        "k_opt": k_opts,
        "dist_null_space": dist_left_null,
    }
