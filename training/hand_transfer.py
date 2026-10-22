"""Scripted Cartesian reference for a nominal right-hook rung transfer.

IK operates on separate scratch data, never on the live simulation state.
This is a full joint-position reference, not a climb-v1 normalized actor.
"""
import mujoco
import numpy as np


def smooth(u):
    u = np.clip(u, 0., 1.)
    return u*u*u*(10.+u*(-15.+6.*u))


class RightHookReference:
    def __init__(self, env, clearance=.055, retreat=.075, rise=.25, initial_clearance=.08):
        self.model = env.model
        self.scratch = mujoco.MjData(env.model)
        self.scratch.qpos[:] = env.initial
        mujoco.mj_kinematics(self.model, self.scratch)
        self.body = self.model.body('right_hook').id
        self.qadr, self.vadr = env.qadr[22:], env.vadr[22:]
        self.bounds = self.model.jnt_range[env.joints[22:]]
        self.start = self.scratch.xpos[self.body].copy()
        dx = rise*np.tan(np.deg2rad(5.))
        self.times = np.array([0., 2., 4., 6., 10., 12., 14., 18.])
        self.offsets = np.array([[0,0,0], [0,0,0], [0,0,initial_clearance],
                                 [-retreat,0,initial_clearance], [-retreat+dx,0,rise+clearance],
                                 [dx,0,rise+clearance], [dx,0,rise], [dx,0,rise]])
        self.labels = ['settle','unload_lift','retreat','rise','approach','seat','load_hold']
        if rise == 0.:
            self.labels[3] = 'free_hold'
        self.preferred = np.array([-2.2,-1.3,.5,.7,1.,.6,-.2])

    def target(self, t):
        i = min(int(np.searchsorted(self.times, t, side='right')-1), len(self.labels)-1)
        i = max(i,0)
        u = smooth((t-self.times[i])/(self.times[i+1]-self.times[i]))
        return self.start+(1-u)*self.offsets[i]+u*self.offsets[i+1], self.labels[i]

    def solve(self, qpos, target, iterations=80):
        m, d = self.model, self.scratch
        # Actual base and other joints; warm start arm from last reference.
        arm = d.qpos[self.qadr].copy()
        d.qpos[:] = qpos
        d.qpos[self.qadr] = arm
        jp, jr = np.zeros((3,m.nv)), np.zeros((3,m.nv))
        for _ in range(iterations):
            mujoco.mj_kinematics(m,d)
            mujoco.mj_comPos(m,d)
            rot = np.zeros(3)
            mujoco.mju_quat2Vel(rot,d.xquat[self.body],1.)
            error = np.r_[target-d.xpos[self.body], -.3*rot]
            if np.linalg.norm(error) < 2e-5:
                break
            mujoco.mj_jacBody(m,d,jp,jr,self.body)
            jac = np.vstack([jp,.3*jr])[:,self.vadr]
            inverse = jac.T@np.linalg.solve(jac@jac.T+1e-5*np.eye(6),np.eye(6))
            delta = inverse@error
            # Use redundant DOF to avoid a wrist-roll limit while lifting.
            delta += .01*(np.eye(7)-inverse@jac)@(self.preferred-d.qpos[self.qadr])
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr]+np.clip(delta,-.04,.04),
                                       self.bounds[:,0],self.bounds[:,1])
        mujoco.mj_kinematics(m,d)
        rot = np.zeros(3)
        mujoco.mju_quat2Vel(rot,d.xquat[self.body],1.)
        return d.qpos[self.qadr].copy(), float(np.linalg.norm(target-d.xpos[self.body])), float(np.linalg.norm(rot))
