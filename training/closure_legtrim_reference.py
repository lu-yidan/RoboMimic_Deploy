"""Two bounded knee/ankle reference commands over the fixed closure controller.

Only desired left knee and ankle-pitch angles change. The original 26D actor has
0.015 rad residual authority; these explicitly add up to 0.003 rad. Motor and
actual joint limits stay unchanged; this is a different controller authority. No simulator state, force,
actuator bound, task reference or acceptance criterion is modified. The filter
persists across the existing controller switches and runs at controller cadence.
"""
import numpy as np

SCALES = np.array([.003, .003], dtype=float)


def install_adaptive_reference(base, fixed_modifier, *, geometry_gain, record=False):
    if len(base.ends) != 1 or abs(float(base.ends[0])-113.66) > 1e-7:
        raise ValueError('Single 113.66-second cycle required')
    if not np.isfinite(geometry_gain) or not 0. <= geometry_gain <= 1.:
        raise ValueError('Finite geometry gain in [0,1] required')
    e=base.env
    names=[e.model.joint(int(e.joints[i])).name for i in (3,4)]
    if names != ['left_knee_joint','left_ankle_pitch_joint']:
        raise ValueError('Unexpected left-leg joint mapping')
    qadr=e.qadr[3:5];vadr=e.vadr[3:5]
    state=dict(command=np.zeros(2),residual=np.zeros(2),velocity=np.zeros(2),last_time=None,trace=[])

    def set_action(action):
        value=np.asarray(action,dtype=float)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError('Two finite reference commands required')
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
            q[qadr]+=state['residual'];v[vadr]+=state['velocity']
            return _original(q,v,active)

        controller.command=command
    state['set_action']=set_action
    return state
