# %%
from functools import partial

import numpy as np
import src.simulations.psweep as ps
from src.bcf.boosted_control_function_2 import BCF, OLS
from src.simulations.oracle_methods import CausalFunction, ConstantFunc, IMPFunction
from src.simulations.parameter_grids import full_grid
from src.simulations.simulations_funcs import run_simulation

# Constants:
DRY_RUN = False
N_WORKERS = 80
DATABASE_DIR = "../results/output_data/"
DATABASE_NAME = "simulations_02-bcf.pk"
RESULT_NAME = "simulations_02-bcf.csv"

SEED = 17837797101191184914958285

METHODS = [
    ("OLS", OLS()),
    ("BCF", BCF(n_exog=10, continuous_mask=np.repeat(True, 10))),
    ("Causal", CausalFunction()),
    ("IMP", IMPFunction()),
    ("ConstFunc", ConstantFunc()),
]

PARAMS = {
    "n_reps": range(50),
    "n": [1000],
    "p": [10],
    "p_effective": [1, 3, 5],
    "tree_depth": [3],
    "r": [10],
    "q": [1, 5, 7, 10],
    "sd_y": [0.1],
}

PARAMS_2 = ps.pgrid(
    ps.plist("interv_strength", range(1, 11)), ps.plist("gamma_norm", [1, 1.5, 2])
)


# %%
# Function definitions
def main():
    # create params grid
    params_simul = ps.pgrid(full_grid(PARAMS, SEED), PARAMS_2)

    # partialise `run_simulation` function
    simul = partial(run_simulation, methods=METHODS)

    # run simulations in parallel
    ps.run_local(
        simul,
        params_simul,
        database_dir=DATABASE_DIR,
        database_basename=DATABASE_NAME,
        simulate=DRY_RUN,
        backup=False,
        skip_dups=True,
        poolsize=N_WORKERS,
    )

    # save to csv
    df = ps.df_read(f"{DATABASE_DIR}{DATABASE_NAME}")
    df.explode(["method_names", "MSE"]).to_csv(
        f"{DATABASE_DIR}{RESULT_NAME}", index=False
    )


# %%
if __name__ == "__main__":
    print("Simulations: sweep through set of params and evaluate BCF vs OLS.")
    main()
    print("Done")
