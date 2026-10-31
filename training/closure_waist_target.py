"""Diagnostic expansion: three bounded waist reference residuals at closure.

This adds command authority outside the published 26D actor. It does not change
physical state, actuators, native limits, contact model or acceptance reference.
Original controller objects/gain scheduling remain intact. Not a 26D PPO gain.
"""
import numpy as np


def install_waist_target(base,bias,*,geometry_gain=1.,ramp_seconds=1.,record=False):
    bias=np.asarray(bias,dtype=float)
    if bias.shape!=(3,) or not np.isfinite(bias).all() or abs(bias).max()>1.:
        raise ValueError('Three finite normalized waist commands required')
    if not 0.<=geometry_gain<=1. or not 0.<ramp_seconds<=10.:
        raise ValueError('Invalid geometry gain or ramp')
    e=base.env
    names=[e.model.joint(int(e.joints[i])).name for i in (12,13,14)]
    if names!=['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']:
        raise ValueError('Unexpected waist joint mapping')
    qadr=e.qadr[12:15];vadr=e.vadr[12:15]
    state=dict(residual=np.zeros(3),velocity=np.zeros(3),last_time=None,trace=[],maximum_offset_rad=0.)
    for controller in (base.baseline,base.contact):
        original=controller.command
        def command(qref,vref,active,_original=original):
            t=float(base.env.data.time)
            u=float(np.clip((t-99.66)/ramp_seconds,0.,1.));target=.015*geometry_gain*u*u*(3.-2.*u)*bias
            if not target.any() and not state['residual'].any():return _original(qref,vref,active)
            if state['last_time']!=t:
                old=state['residual'].copy();state['residual']+=.05*(target-old)
                state['velocity']=(state['residual']-old)/.005;state['last_time']=t
                state['maximum_offset_rad']=max(state['maximum_offset_rad'],float(abs(state['residual']).max()))
                if record:state['trace'].append(np.r_[t,target,state['residual'],state['velocity']].tolist())
            q=qref.copy();v=vref.copy();q[qadr]+=state['residual'];v[vadr]+=state['velocity']
            return _original(q,v,active)
        controller.command=command
    return state
