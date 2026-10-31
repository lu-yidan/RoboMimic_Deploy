"""Open-loop loading ablation matched to the observed T12 pilot delays.

This controller never reads contact feedback. It is a diagnostic comparator,
not an adopted controller and not a contact-confirmation mechanism.
"""
import hashlib
import inspect
import textwrap
import numpy as np
from training.climb_cycle import CycleTorqueController
from training.probe_hook_loading import capped_command, ORIGINAL_COMMAND_SHA256

PILOT_DELAYS={'left_hook':.090,'right_hook':.085}


def install_timed_hook_loading(base,probe_force_n=15.,ramp_seconds=.5,delays=None):
    delays=dict(PILOT_DELAYS if delays is None else delays)
    if set(delays)!=set(PILOT_DELAYS) or not all(np.isfinite(v) and 0<=v<=.2 for v in delays.values()):
        raise ValueError('Expected bounded per-hook open-loop delay')
    if not (0<probe_force_n<100 and .1<=ramp_seconds<=1):raise ValueError('Invalid loading bounds')
    source=textwrap.dedent(inspect.getsource(CycleTorqueController.command))
    if hashlib.sha256(source.encode()).hexdigest()!=ORIGINAL_COMMAND_SHA256:
        raise ValueError('Original QP changed; review the shared capped command')
    original=base.baseline.command
    state=dict(previous=np.ones(4,dtype=bool),hooks={},trace=[])
    labels={2:'left_hook',3:'right_hook'}
    def command(qref,vref,active):
        t=float(base.env.data.time)
        cycle=min(int(np.searchsorted(base.ends,t,side='right')),len(base.ends)-1)
        active=np.asarray(active,dtype=bool)
        if cycle==0:
            state['previous']=active.copy();return original(qref,vref,active)
        caps=np.full(4,350.)
        for index,group in labels.items():
            if not active[index]:state['hooks'].pop(index,None);continue
            if not state['previous'][index]:state['hooks'][index]=dict(start=t,released=None)
            hook=state['hooks'].get(index)
            if hook is None:continue
            if hook['released'] is None and t-hook['start']>=delays[group]-1e-8:hook['released']=t
            if hook['released'] is None:caps[index]=probe_force_n/2.
            else:
                u=float(np.clip((t-hook['released'])/ramp_seconds,0,1));w=u*u*(3-2*u)
                caps[index]=(probe_force_n+(700.-probe_force_n)*w)/2.
        state['previous']=active.copy()
        limited=bool(np.any(caps[active]<350.-1e-10))
        torque,details=capped_command(base.baseline,qref,vref,active,caps) if limited else original(qref,vref,active)
        if limited:
            state['trace'].append(dict(time_s=t,cycle=cycle+1,active=active.tolist(),
                caps_per_contact_n=caps.tolist(),hooks={labels[k]:dict(v) for k,v in state['hooks'].items()},
                force_plan=details['force_plan'],torque_nm=np.asarray(torque).tolist(),
                base_equilibrium_residual=details['base_equilibrium_residual'],controller_type='timed_open_loop'))
        return torque,details
    base.baseline.command=command
    return state
