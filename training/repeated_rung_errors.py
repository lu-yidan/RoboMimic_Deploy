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
    # Validate all requested rows before any mutation.
    for row,vector in offsets.items():
        delta=np.asarray(vector,dtype=float)
        if row not in ('rung_01','rung_02','rung_05','rung_06') or delta.shape!=(3,) or not np.isfinite(delta).all() or np.max(abs(delta))>.02 or delta[1]!=0:
            raise ValueError('Invalid two-cycle target-row offset')
    changes = []
    for name, vector in offsets.items():
        if name not in ('rung_01','rung_02','rung_05','rung_06'):
            raise ValueError('Only declared two-cycle target rows supported')
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

