"""Fade existing closure references after the first accepted cycle.
 
The first-cycle arithmetic matches the frozen single-cycle installers. Filters
persist across the seam. Only desired references change, never physical state.
"""
import numpy as np
from training.closure_body_target import body_target_offset as single_cycle_body_offset


def handoff_fade(time_s, first_end, seconds):
    if not np.isfinite(seconds) or not 1. <= seconds <= 10.:
        raise ValueError('Reference fade must last one to ten seconds')
    if time_s <= first_end+1e-7:
        return 1., 0.
    u = float(np.clip((time_s-first_end)/seconds, 0., 1.))
    value = 1.-(10*u**3-15*u**4+6*u**5)
    derivative = -30*u*u*(1-u)*(1-u)/seconds if 0. < u < 1. else 0.
    return value, derivative


def body_target_offset(time_s, drop_m, *, first_end=113.66, fade_seconds=6.):
    position, velocity = single_cycle_body_offset(time_s, drop_m)
    if time_s <= first_end+1e-7:
        return position, velocity
    weight, rate = handoff_fade(time_s, first_end, fade_seconds)
    return position*weight, velocity*weight+position*rate


def install_waist_target(base,bias,*,geometry_gain=1.,ramp_seconds=1.,record=False,fade_seconds=6.):
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
            if t > float(base.ends[0])+1e-7:
                target *= handoff_fade(t, float(base.ends[0]), fade_seconds)[0]
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



def install_body_target(base, drop_m, *, fade_seconds=6.):
    body_target_offset(0.,drop_m)
    for controller in (base.baseline,base.contact):
        original=controller.command

        def command(qref,vref,active,_original=original):
            dz,dv=body_target_offset(float(base.env.data.time),drop_m,first_end=float(base.ends[0]),fade_seconds=fade_seconds)
            if dz==0. and dv==0.:return _original(qref,vref,active)
            q=qref.copy();v=vref.copy();q[2]+=dz;v[2]+=dv
            return _original(q,v,active)

        controller.command=command



SCALES = np.array([.003, .003, .003, .00015, .001], dtype=float)


def install_adaptive_reference(base, fixed_modifier, *, geometry_gain, record=False, fade_seconds=6.):
    if len(base.ends) != 2 or abs(float(base.ends[0])-113.66) > 1e-7:
        raise ValueError('Two cycles with an unchanged 113.66-second first cycle required')
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
    state=dict(command=np.zeros(5),residual=np.zeros(5),velocity=np.zeros(5),last_time=None,trace=[])

    def set_action(action):
        value=np.asarray(action,dtype=float)
        if value.shape != (5,) or not np.isfinite(value).all():
            raise ValueError('Five finite reference commands required')
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
                if t > float(base.ends[0])+1e-7:
                    target *= handoff_fade(t, float(base.ends[0]), fade_seconds)[0]
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
            q[0]+=state['residual'][4];v[0]+=state['velocity'][4]
            return _original(q,v,active)

        controller.command=command
    state['set_action']=set_action
    return state


