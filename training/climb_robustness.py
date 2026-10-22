"""Declared scene errors and contact-based release interlock for cycle experiments."""
from collections import deque
import hashlib
import json
import mujoco
import numpy as np

GROUPS = ('left_foot', 'right_foot', 'left_hook', 'right_hook')
BODIES = ('left_ankle_roll_link', 'right_ankle_roll_link', 'hook', 'right_hook')


def apply_rung_offsets(env, offsets):
    """Move future bars and their mounting geometry; never mutate the frozen bundle."""
    changes = []
    for name, vector in offsets.items():
        if name not in ('rung_01', 'rung_05'):
            raise ValueError('Only future first-cycle target rows 01 and 05 are supported')
        delta = np.asarray(vector, dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all() or np.max(abs(delta)) > .02 or delta[1] != 0:
            raise ValueError('Offsets must be finite X/Z translations within 20 mm')
        for i in range(env.model.ngeom):
            geom = env.model.geom(i)
            if geom.name == name or geom.name.startswith(name + '_'):
                if env.model.geom_bodyid[i] != 0:
                    raise ValueError('Expected fixed world geometry')
                before = env.model.geom_pos[i].copy()
                env.model.geom_pos[i] += delta
                changes.append(dict(name=geom.name, before=before.tolist(), after=env.model.geom_pos[i].tolist()))
        sid = env.model.site(name + '_target').id
        env.model.site_pos[sid] += delta
    # Static world BVH bounds are not refreshed by mj_forward or mj_setConst.
    # Recompile the declared geometry to obtain correct bounds, preserving all
    # live state and the original joint/actuator indexing. Never call mj_setConst
    # here: it also changes qpos, which would invalidate dynamic continuity.
    if offsets:
        spec=mujoco.MjSpec.from_file(str(env.bundle/'model.xml'))
        for geom in spec.geoms:
            for name,vector in offsets.items():
                if geom.name==name or geom.name.startswith(name+'_'):
                    geom.fromto=np.asarray(geom.fromto)+np.tile(vector,2)
        compiled=spec.compile()
        if not np.array_equal(compiled.bvh_child,env.model.bvh_child) or not np.array_equal(compiled.bvh_nodeid,env.model.bvh_nodeid):
            raise ValueError('Scene error changed BVH topology; rebuild the entire model instead')
        np.testing.assert_allclose(compiled.geom_pos,env.model.geom_pos,atol=1e-12,rtol=0.)
        env.model.bvh_aabb[:]=compiled.bvh_aabb
    mujoco.mj_forward(env.model, env.data)
    return dict(offsets={k:list(v) for k,v in offsets.items()}, changed_geoms=changes,
                geometry_sha256=hashlib.sha256(env.model.geom_pos.tobytes()+env.model.site_pos.tobytes()+env.model.bvh_aabb.tobytes()).hexdigest(),
                static_bvh='Recompiled from displaced fixed geometry; topology verified identical',
                note='In-memory fixed-bar and bracket translation. Nominal controller reference unchanged; no perception correction.')


class ReleaseInterlock:
    """Require continuous load, correct rod identity and low contact-point speed."""
    def __init__(self, env, duration=.25, minimum_n=10., maximum_speed=.02):
        self.env = env
        self.duration = duration
        self.minimum_n = minimum_n
        self.maximum_speed = maximum_speed
        self.history = deque()
        self.expected = ['rung_00','rung_00','rung_04','rung_04']
        self.previous_active = np.ones(4, dtype=bool)
        self.events = []

    def sample(self, t):
        metrics = self.env.contact_metrics()
        valid = []
        for i, group in enumerate(GROUPS):
            contacts = [c for c in metrics['contacts'] if c['group']==group and np.linalg.norm(c['force_on_robot_world_n'])>1.]
            force = sum(c['force_on_robot_world_n'][2] for c in contacts if c['external_geom']==self.expected[i])
            body = self.env.model.body(BODIES[i]).id
            local = np.array([.09,0,-.035] if i<2 else [0,0,.022])
            point = self.env.data.xpos[body]+self.env.data.xmat[body].reshape(3,3)@local
            jac = np.zeros((3,self.env.model.nv))
            mujoco.mj_jac(self.env.model,self.env.data,jac,None,point,body)
            speed = float(np.linalg.norm(jac@self.env.data.qvel))
            valid.append(bool(force>=self.minimum_n and contacts and all(c['external_geom']==self.expected[i] for c in contacts) and speed<=self.maximum_speed))
        self.history.append((float(t),valid))
        while len(self.history)>1 and self.history[1][0] < t-self.duration-1e-8:
            self.history.popleft()

    def allow(self, t, active):
        active = np.asarray(active,dtype=bool)
        released = np.flatnonzero(self.previous_active & ~active)
        landed = np.flatnonzero(~self.previous_active & active)
        for i in landed:
            self.expected[i] = f'rung_{int(self.expected[i][-2:])+1:02d}'
        if len(released):
            remaining = np.flatnonzero(active)
            covered = bool(self.history) and t-self.history[0][0]>=self.duration-1e-7
            stable = covered and all(all(v[i] for i in remaining) for _,v in self.history)
            self.events.append(dict(time_s=float(t),released=[GROUPS[i] for i in released],
                                    allowed=bool(stable),expected_rungs=self.expected.copy(),window_s=self.duration))
            if not stable:
                return False
        self.previous_active = active.copy()
        return True
