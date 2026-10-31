"""Experimental bounded body-height command during terminal leg feedback.

This adds controller authority beyond the frozen actor's 26 joint actions.
It changes desired acceleration through qref/vref, never simulator state or
the reference used by the endpoint acceptance checks.
"""
import numpy as np


def body_target_offset(time_s, drop_m):
    if not np.isfinite(drop_m) or not 0. <= drop_m <= .015:
        raise ValueError('Body target drop must be within [0, 0.015] m')
    u=float(np.clip((time_s-99.66)/3.,0.,1.))
    position=-drop_m*u*u*(3.-2.*u)
    velocity=-drop_m*2.*u*(1.-u) if 0.<u<1. else 0.
    return position,velocity


def install_body_target(base, drop_m):
    body_target_offset(0.,drop_m)
    for controller in (base.baseline,base.contact):
        original=controller.command

        def command(qref,vref,active,_original=original):
            dz,dv=body_target_offset(float(base.env.data.time),drop_m)
            if dz==0. and dv==0.:return _original(qref,vref,active)
            q=qref.copy();v=vref.copy();q[2]+=dz;v[2]+=dv
            return _original(q,v,active)

        controller.command=command
