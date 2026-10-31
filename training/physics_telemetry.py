"""Read-only per-physics-step telemetry without replacing controller objects."""
import numpy as np
import mujoco
from training.climb_robustness import GROUPS


def install_physics_telemetry(base):
    e=base.env;m=e.model;d=e.data
    count=int(np.ceil(base.horizon/m.opt.timestep))+10
    data=dict(time_s=np.empty(count),solver_qacc=np.empty((count,m.nv)),
        post_velocity=np.empty((count,6)),group_force_world_n=np.empty((count,len(GROUPS),3)),
        motor_torque_nm=np.empty((count,len(e.motors))))
    state=dict(count=0,data=data,groups=list(GROUPS))
    original=base.physics_check
    def check(qref,phase,controller):
        reason=original(qref,phase,controller)
        i=state['count']
        if i>=count:raise RuntimeError('Telemetry capacity exceeded')
        data['time_s'][i]=d.time;data['solver_qacc'][i]=d.qacc
        data['post_velocity'][i]=d.qvel[:6];data['motor_torque_nm'][i]=d.ctrl[e.motors]
        forces=np.zeros((len(GROUPS),3))
        for ci,contact in enumerate(d.contact):
            ba,bb=(int(m.geom_bodyid[g]) for g in (contact.geom1,contact.geom2))
            ia,ib=ba in e.robot_bodies,bb in e.robot_bodies
            if ia==ib:continue
            body=ba if ia else bb
            if body not in e.contact_groups:continue
            group=e.contact_groups[body]
            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
            forces[GROUPS.index(group)]+=(-1. if ia else 1.)*(wrench[:3]@contact.frame.reshape(3,3))
        data['group_force_world_n'][i]=forces
        state['count']+=1
        return reason
    base.physics_check=check
    return state
