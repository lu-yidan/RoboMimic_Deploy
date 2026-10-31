"""Reconstruct approach kinematics only at saved poses before force peaks."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.physical_v2_cycle_env import PhysicalV2CycleEnv
from tools.validate_swing_reference import read,write,sha


def ordered_distance(model,data,first,second,cutoff=.05):
    """Call with ordered types and restore caller order for witness points.

    MuJoCo 3.3.7's native-CCD branch returns before undoing its internal
    geom-type sorting. Sorting explicitly avoids relying on that branch.
    This read-only query does not alter the collision model or dynamics.
    """
    swap=int(model.geom_type[first])>int(model.geom_type[second])
    a,b=(second,first) if swap else (first,second)
    witness=np.zeros(6)
    distance=float(mujoco.mj_geomDistance(model,data,a,b,cutoff,witness))
    if swap:witness=np.r_[witness[3:],witness[:3]]
    return distance,witness


def one(run,events):
    m=read(run/'manifest.json');a=read(run/'v2_multicycle_audit.json');parent=read(Path(m['parent_experiment'])/'manifest.json')
    assert a['passed'] and a['manifest_sha256']==sha(run/'manifest.json')
    assert a['trajectory_sha256']==sha(run/'trajectory.npz')
    record=read(run/('hook_admission_record.json' if m.get('contact_admission') else 'telemetry_audit.json'))
    if not m.get('contact_admission'):assert record['passed']
    assert record['trajectory_sha256']==sha(run/'trajectory.npz')
    assert record['telemetry_sha256']==sha(run/'physics_telemetry.npz')
    with np.load(run/'trajectory.npz') as z:x={k:z[k] for k in ('qpos','qvel','time_s')}
    with np.load(run/'physics_telemetry.npz') as z:times=z['time_s'];forces=z['group_force_world_n']
    base=PhysicalV2CycleEnv(plan=m['plan_path'],cases={m['case']:m['offsets']},
        normal_bias_y_deg=parent['normal_bias_y_deg'],memory_enabled=parent['force_history_memory'])
    base.reset(options={'case':m['case']});e=base.env;model=e.model;data=e.data
    rows=[]
    try:
        target=model.geom('rung_06').id;assert model.geom_bodyid[target]==0
        for event in events:
            g=event['group'];start=event['start_s'];j=record['groups'].index(g)
            selected=np.flatnonzero((times>=start)&(times<=start+.2));peak=selected[np.argmax(forces[selected,j,2])]
            peak_time=float(times[peak]);saved=int(np.searchsorted(x['time_s'],peak_time,side='left')-1)
            data.qpos[:]=x['qpos'][saved];data.qvel[:]=x['qvel'][saved];mujoco.mj_forward(model,data)
            candidates=[i for i in range(model.ngeom) if e.contact_groups.get(int(model.geom_bodyid[i]))==g
                        and (model.geom_contype[i] or model.geom_conaffinity[i])]
            distances=[]
            for geom in candidates:
                distance,witness=ordered_distance(model,data,geom,target)
                distances.append((distance,geom,witness))
            gap,geom,witness=min(distances,key=lambda r:r[0]);closing=None;finite=None;error=None
            if 1e-8<gap<.05:
                direction=witness[3:]-witness[:3];direction/=np.linalg.norm(direction)
                J=np.zeros((3,model.nv));mujoco.mj_jac(model,data,J,None,witness[:3],int(model.geom_bodyid[geom]))
                closing=float(direction@(J@data.qvel));dist=[]
                for dt in (-1e-6,1e-6):
                    data.qpos[:]=x['qpos'][saved];mujoco.mj_integratePos(model,data.qpos,x['qvel'][saved],dt)
                    mujoco.mj_forward(model,data)
                    dist.append(ordered_distance(model,data,geom,target)[0])
                finite=-(dist[1]-dist[0])/2e-6;error=abs(finite-closing)
                if error>=1e-4:
                    raise AssertionError(f'Ambiguous distance derivative: {g}, gap={gap}, Jacobian={closing}, finite={finite}, error={error}; do not interpret')
            rows.append(dict(group=g,activation_s=start,actual_vertical_peak_n=float(forces[peak,j,2]),
                actual_peak_time_s=peak_time,saved_state_time_s=float(x['time_s'][saved]),
                time_before_peak_ms=(peak_time-float(x['time_s'][saved]))*1000,
                closest_hook_geom=model.geom(geom).name,signed_gap_mm=gap*1000,
                approach_speed_m_s=closing,distance_derivative_approach_speed_m_s=finite,derivative_error_m_s=error))
    finally:base.close()
    return dict(run=str(run),rows=rows,manifest_sha256=sha(run/'manifest.json'),
        trajectory_sha256=sha(run/'trajectory.npz'),telemetry_sha256=record['telemetry_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    events=read(a.candidate/'hook_admission_audit.json')['events'];assert len(events)==2
    report=dict(baseline=one(a.baseline.resolve(),events),candidate=one(a.candidate.resolve(),events),
        source_sha256=sha(__file__),new_physics_steps=0,
        engine_version=mujoco.__version__,witness_order_source='https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_support.c',
        scope='Last saved 50 Hz pose strictly before each first-200-ms vertical-force peak; not instantaneous collision speed. Closest collision-point Jacobian projected toward the fixed rod, checked against a 1 microsecond kinematic distance derivative. No interpolation, contact-law fitting, causal isolation, new controller or dynamic rollout.')
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'summary.json',report)
    (a.output/'analysis_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(report),flush=True)
