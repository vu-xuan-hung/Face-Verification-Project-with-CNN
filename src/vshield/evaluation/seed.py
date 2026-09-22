"""Seed evaluation-side randomness without changing production inference."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
