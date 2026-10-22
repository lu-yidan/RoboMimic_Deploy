"""Reject an infeasible support removal before sending its motor command."""
import numpy as np


def allocate_before_release(controller,qref,vref,active,hold_q,hold_active):
    """Return a feasible command and, on rejection, the candidate failure reason.

    Both allocations use identical native motor bounds and contact constraints.
    No command is applied to MuJoCo here. The caller must freeze its phase and
    retain hold_active if the returned reason is non-null.
    """
    try:
        torque,info=controller.command(qref,vref,active)
        return torque,info,None
    except RuntimeError as rejected:
        try:
            torque,info=controller.command(hold_q,np.zeros_like(vref),hold_active)
        except RuntimeError as hold_failure:
            raise RuntimeError(f'Release rejected ({rejected}); retained-support allocation also failed ({hold_failure})') from hold_failure
        return torque,info,str(rejected)
