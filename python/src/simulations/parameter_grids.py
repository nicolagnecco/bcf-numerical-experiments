import psweep as ps
import numpy as np


def full_grid(params: dict, seed: int):

    param_list = []

    # create plist
    for key, value in params.items():
        param_list.append(ps.plist(key, value))

    # take Cartesian product
    param_grid = ps.pgrid(*param_list)

    # define SeedSequence
    ss = np.random.SeedSequence(seed)
    child_states = ss.spawn(len(param_grid))

    # append them to param_grid list
    param_grid_out = []

    for param, child_state in zip(param_grid, child_states):
        param_grid_out.append(param | {"seed": child_state})

    return param_grid_out
