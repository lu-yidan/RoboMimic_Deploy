"""Attempt a bounded current-pose stop after a tracking allocation failure."""
import numpy as np


def current_pose_hold(controller, active, confirmed):
    """Compute only: never apply torque, modify state, or add a support.

    Confirmation is supplied by the caller's existing time-window guard. This
    function establishes feasibility for one instant, not future hold stability.
    """
    active=np.asarray(active,dtype=bool);confirmed=np.asarray(confirmed,dtype=bool)
    if active.shape!=(4,) or confirmed.shape!=(4,):
        raise ValueError('Expected four support flags')
    retained=active & confirmed
    if np.count_nonzero(retained)<3:
        raise RuntimeError('Fewer than three confirmed scheduled contacts for current-pose hold')
    q=controller.d.qpos.copy();v=np.zeros(controller.m.nv)
    torque,info=controller.command(q,v,retained)
    return torque,info,q,retained
