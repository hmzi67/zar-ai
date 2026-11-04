"""Seed every RNG in use."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError:  # torch only exists with the vision/nlp extras
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
