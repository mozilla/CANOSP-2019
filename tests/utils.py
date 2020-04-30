"""Shared testing utils."""
import random

import numpy as np


def reset_random_seed():
    random.seed(42)
    np.random.seed(42)
