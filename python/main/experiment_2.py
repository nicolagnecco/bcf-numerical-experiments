# %%
# Simulation on rank recovery
from functools import partial

import numpy as np
import src.simulations.psweep as ps
from src.bcf.reduced_rank_regression import RRR
from src.simulations.parameter_grids import full_grid
from src.simulations.simulations_funcs import run_simulation_rank

# Constants:
DRY_RUN = False
N_WORKERS = 8
DATABASE_DIR = "../results/output_data/"
DATABASE_NAME = "experiment_2-nullspace.pk"
RESULT_NAME = "experiment_2-nullspace.csv"

SEED = 881588530


METHODS = [
    (
        "ReducedRankRegression",
        RRR(alpha=0),
        {"alpha": 10 ** np.arange(-5.0, 6)},
        "neg_root_mean_squared_error",
    ),
]

PARAMS = {
    "n_reps": range(50),
    "n": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000],
    "p": [10, 15],
    "r": [10, 15],
    "q": [5],
    "eigengap": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "p_effective": [1],
    "tree_depth": [3],
    "sd_y": [0.1],
    "interv_strength": [1],
    "gamma_norm": [1],
}


# %%
# Function definitions
def main():
    # create params grid
    params_simul = full_grid(PARAMS, SEED)

    # partialise `run_simulation` function
    simul = partial(run_simulation_rank, methods=METHODS)  # type: ignore

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
    df.explode(["k_opt", "dist_null_space"]).to_csv(
        f"{DATABASE_DIR}{RESULT_NAME}", index=False
    )


# %%
if __name__ == "__main__":
    print(
        "Simulations: sweep through set of params and evaluate reduced rank regression."
    )
    main()
    print("Done")
