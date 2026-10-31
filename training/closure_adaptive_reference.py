"""Four bounded feedback commands added to a fixed closure controller.

Only desired waist angles and body height change. No simulator state, force,
actuator bound, task reference or acceptance criterion is modified. The filter
persists across the existing controller switches and runs at controller cadence.
"""
import numpy as np

SCALES = np.array([.003, .003, .003, .00015], dtype=float)


def install_adaptive_reference(base, fixed_modifier, *, geometry_gain, record=False):
    if len(base.ends) != 1 or abs(float(base.ends[0])-113.66) > 1e-7:
        raise ValueError('Single 113.66-second cycle required')
    if not np.isfinite(geometry_gain) or not 0. <= geometry_gain <= 1.:
        raise ValueError('Finite geometry gain in [0,1] required')
    fixed_waist=np.asarray(fixed_modifier['waist_bias'],dtype=float)*.015
    if fixed_waist.shape != (3,) or not np.isfinite(fixed_waist).all() or np.any(abs(fixed_waist)+SCALES[:3]>.015):
        raise ValueError('Combined waist reference would exceed 0.015 rad')
    fixed_drop=float(fixed_modifier['body_drop_m'])
    if not SCALES[3] <= fixed_drop <= .015-SCALES[3]:
        raise ValueError('Combined body drop must remain within [0,0.015] m')
    e=base.env
    names=[e.model.joint(int(e.joints[i])).name for i in (12,13,14)]
    if names != ['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']:
        raise ValueError('Unexpected waist joint mapping')
    qadr=e.qadr[12:15];vadr=e.vadr[12:15]
    state=dict(command=np.zeros(4),residual=np.zeros(4),velocity=np.zeros(4),last_time=None,trace=[])

    def set_action(action):
        value=np.asarray(action,dtype=float)
        if value.shape != (4,) or not np.isfinite(value).all():
            raise ValueError('Four finite reference commands required')
        state['command'][:]=np.clip(value,-1.,1.)

    for controller in (base.baseline,base.contact):
        original=controller.command

        def command(qref,vref,active,_original=original):
            t=float(e.data.time)
            if state['last_time'] != t:
                if state['last_time'] is not None and abs(t-state['last_time']-.005)>1e-7:
                    raise ValueError('Adaptive reference requires uninterrupted 200 Hz controller cadence')
                u=float(np.clip((t-99.66)/3.,0.,1.));ramp=u*u*(3.-2.*u)
                target=SCALES*geometry_gain*ramp*state['command']
                previous=state['residual'].copy()
                state['residual']+=.05*(target-previous)
                state['velocity']=(state['residual']-previous)/.005
                state['last_time']=t
                if record:
                    state['trace'].append(np.r_[t,state['command'],target,state['residual'],state['velocity']].tolist())
            if not state['residual'].any() and not state['velocity'].any():
                return _original(qref,vref,active)
            q=qref.copy();v=vref.copy()
            q[qadr]+=state['residual'][:3];v[vadr]+=state['velocity'][:3]
            q[2]+=state['residual'][3];v[2]+=state['velocity'][3]
            return _original(q,v,active)

        controller.command=command
    state['set_action']=set_action
    return state
