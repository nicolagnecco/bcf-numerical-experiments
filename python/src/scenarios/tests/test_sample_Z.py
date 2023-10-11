import pytest
import numpy as np
from src.scenarios.generate_x import sample_Z

# Constants
N = 100
P = 5
R = 2

# Tests
def test_sample_Z():
    Z1 = sample_Z(N, P, R)
    assert np.unique(Z1[1]).size == R + 1