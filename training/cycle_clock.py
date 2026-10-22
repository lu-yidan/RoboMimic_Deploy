"""Cycle boundaries use the same half-open intervals as reference interpolation."""
import numpy as np


def cycle_index(time_s,cycle_ends):
    return int(np.searchsorted(cycle_ends,float(time_s),side='right'))
