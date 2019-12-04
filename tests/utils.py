"""Shared testing utils."""
import random

import numpy as np
from decouple import config

HOSTNAME = config("FLDP_HOST", default="127.0.0.1")
PORT = config("FLDP_PORT", default=8000)


def reset_random_seed():
    random.seed(42)
    np.random.seed(42)
