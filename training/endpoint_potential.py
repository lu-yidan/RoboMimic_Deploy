"""Bounded endpoint potential for a separately versioned reward experiment.

Reference: Ng, Harada and Russell (1999), Policy invariance under reward
transformations. https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
This module is not used by the existing v1/v2 training environments.
"""
import math


def endpoint_potential(error_m, ramp, *, terminal=False, weight=4., length_scale=.01, cap=9.):
    if not all(math.isfinite(value) for value in (weight, length_scale, cap)) or weight < 0 or length_scale <= 0 or cap <= 0:
        raise ValueError('Finite nonnegative weight and positive scale/cap required')
    if terminal:
        return 0.
    if not math.isfinite(error_m) or error_m < 0 or not math.isfinite(ramp) or not 0 <= ramp <= 1:
        raise ValueError('Finite nonnegative error and ramp in [0,1] required')
    return -weight*ramp*min((error_m/length_scale)**2, cap)


def potential_reward(previous, current, gamma):
    if not all(math.isfinite(value) for value in (previous, current, gamma)) or not 0 < gamma <= 1:
        raise ValueError('Finite potentials and discount in (0,1] required')
    return gamma*current-previous
