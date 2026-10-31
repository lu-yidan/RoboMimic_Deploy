"""Compile complete fixed-rod variants at episode reset, including the BVH.

The adopted offset helper remains unchanged. New experiments must explicitly
select these environment classes and freeze this source. No model is replaced
during an episode. Native dynamics, torque controllers and limits are inherited.
"""
from collections import deque
import hashlib
import inspect
import textwrap
import gymnasium as gym
import mujoco
import numpy as np
from training.full_cycle_residual_env import FullCycleResidualEnv as OriginalEnv
from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv
from training.physical_v2_cycle_env import PhysicalV2CycleEnv
from training.climb_cycle import CycleTorqueController
from training.periodic_force_history_controller import PeriodicForceHistoryController

ORIGINAL_RESET_SHA256='3f5f225677e7776d0be2016503a93f91f1d5dee7ce264febd141657026ea30c5'
ROWS=('rung_01','rung_02','rung_05','rung_06')


def compiled_variant(base,offsets):
    """Return a complete compiler-produced model with unchanged robot schema."""
    e=base.env;old=e.model
    for name,vector in offsets.items():
        delta=np.asarray(vector,dtype=float)
        if name not in ROWS or delta.shape!=(3,) or not np.isfinite(delta).all() or abs(delta).max()>.02 or delta[1]!=0:
            raise ValueError('Expected declared X/Z rod translations within 20 mm')
    spec=mujoco.MjSpec.from_file(str(e.bundle/'model.xml'))
    moved=[]
    for geom in spec.geoms:
        for rod,vector in offsets.items():
            if geom.name==rod or geom.name.startswith(rod+'_'):
                if old.geom_bodyid[old.geom(geom.name).id]!=0:
                    raise ValueError('Only world-fixed rods and mounts may move')
                geom.fromto=np.asarray(geom.fromto)+np.tile(vector,2)
                moved.append(geom.name)
    for rod,vector in offsets.items():
        site=spec.site(rod+'_target')
        if old.site_bodyid[old.site(site.name).id]!=0:
            raise ValueError('Only world-fixed targets may move')
        site.pos=np.asarray(site.pos)+vector
    model=spec.compile()
    # A fresh model/data pair allows BVH node permutations. Structural robot
    # indices, actuator authority and all non-geometric arrays must stay fixed.
    assert model.names==old.names
    permitted={'geom_pos','site_pos','geom_quat','geom_size','geom_aabb','geom_rbound',
               'bvh_aabb','bvh_child','bvh_depth','bvh_nodeid'}
    checked=0
    for name in dir(old):
        before=getattr(old,name)
        if not isinstance(before,np.ndarray):continue
        after=getattr(model,name)
        assert before.shape==after.shape and before.dtype==after.dtype,name
        if name not in permitted:assert np.array_equal(before,after),name
        checked+=1
    for field in ('nq','nv','nu','nbody','njnt','ngeom','nsite','neq','nbvh','nbvhstatic','nbvhdynamic'):
        assert getattr(old,field)==getattr(model,field),field
    expected_geom=base.nominal_geometry[0].copy();expected_site=base.nominal_geometry[1].copy()
    for rod,vector in offsets.items():
        for i in range(model.ngeom):
            name=model.geom(i).name or ''
            if name==rod or name.startswith(rod+'_'):expected_geom[i]+=vector
        expected_site[model.site(rod+'_target').id]+=vector
    np.testing.assert_allclose(model.geom_pos,expected_geom,rtol=0,atol=1e-12)
    np.testing.assert_allclose(model.site_pos,expected_site,rtol=0,atol=1e-12)
    # Translating capsule endpoints can alter compiled sizes/quaternions at
    # floating-point roundoff. Bound and report this rather than claiming
    # bitwise equality to the older in-memory-offset implementation.
    roundoff={name:float(abs(getattr(model,name)-getattr(old,name)).max(initial=0))
              for name in ('geom_quat','geom_size','geom_aabb','geom_rbound')}
    assert max(roundoff.values())<1e-12
    digest=hashlib.sha256()
    for name in ('geom_pos','site_pos','bvh_aabb','bvh_child','bvh_depth','bvh_nodeid'):
        digest.update(getattr(model,name).tobytes())
    return model,dict(offsets={k:list(v) for k,v in offsets.items()},moved_geoms=moved,
        geometry_sha256=digest.hexdigest(),scene_compiler='complete_mjspec_model_at_reset_v1',
        model_arrays_checked=checked,geometry_roundoff=roundoff,
        static_bvh='Entire model compiled by MuJoCo; node identities and topology are retained from that model',
        note='Fresh model/data only at episode reset. No in-episode state overwrite or weakened collision guard.')


def reset_compiled_geometry(base,*,seed=None,options=None):
    source=textwrap.dedent(inspect.getsource(OriginalEnv.reset))
    if hashlib.sha256(source.encode()).hexdigest()!=ORIGINAL_RESET_SHA256:
        raise ValueError('Original episode initialization changed; review compiled reset before use')
    gym.Env.reset(base,seed=seed)
    names=list(base.cases)
    base.case=(options or {}).get('case',names[int(base.np_random.integers(len(names)))])
    if base.case not in base.cases:raise ValueError('Unknown declared geometry case')
    e=base.env
    if e.renderer is not None:raise ValueError('Close the renderer before compiling a new episode scene')
    model,base.variant=compiled_variant(base,base.cases[base.case])
    e.model=model;e.data=mujoco.MjData(model)
    e.qadr=model.jnt_qposadr[e.joints].copy();e.vadr=model.jnt_dofadr[e.joints].copy()
    e.limits=model.actuator_ctrlrange[e.motors].copy()
    e.reset();m,d=e.model,e.data
    e.initial=base.plan['qpos'][0].copy();d.qpos[:]=e.initial
    mujoco.mj_forward(m,d)
    base.baseline=CycleTorqueController(e,joint_kp=100.,leg_kp=100.,torque_weight=10000.)
    base.contact=PeriodicForceHistoryController(e,joint_kp=100.,leg_kp=100.,torque_weight=10000.,
        normal_bias_y_deg=base.normal_bias,memory_enabled=base.memory_enabled,cycle_ends=base.ends)
    base.controller_cycle=0;base.transfer_entered=False;base.last_phase=None
    base.phase_start=0.;base.phase_gain=100.
    base.residual=np.zeros(26);base.previous_action=np.zeros(26);base.last_torque=np.zeros(29)
    base.steps=0;base.finished=False
    base.peaks=dict(other=0.,self=0.,flight=0.,base=0.,torque=0.)
    base.minimum_margin=float('inf');base.tail=deque();base.cycle_results=[]
    base.warning_start=d.warning.number.copy();base.last_mode='fixed_contact_points'
    return base.observation(),base.info('reset',False)


class CompiledStrictCycleEnv(StrictEnv):
    def reset(self,**kwargs):
        self.strict_tail=deque()
        return reset_compiled_geometry(self,**kwargs)


class CompiledPhysicalV2CycleEnv(PhysicalV2CycleEnv):
    def reset(self,**kwargs):
        self.physical_reset_count=getattr(self,'physical_reset_count',0)+1
        self.v2_tail=[];self.v2_endpoint_records=[];self.strict_tail=deque()
        return reset_compiled_geometry(self,**kwargs)
