"""Additional online-reference checks; preserve original cycle acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    z=dict(np.load(a.run/'online_reference.npz'));plan=dict(np.load(a.run/'reference.npz'))
    rows=json.loads((a.run/'landing_log.json').read_text());manifest=json.loads((a.run/'manifest.json').read_text())
    summary=json.loads((a.run/'summary.json').read_text());e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2')
    scratch=mujoco.MjData(e.model);nominal=mujoco.MjData(e.model)
    counts=len(z['time_s']);assert counts>0
    rows=rows[:counts]
    bounds=identity=velocity=True;position_error=0.;orientation_error=0.;previous=np.zeros(6)
    other=np.setdiff1d(np.arange(e.model.nq),e.qadr[:6])
    for i,t in enumerate(z['time_s']):
        j=min(max(np.searchsorted(plan['time'],t,side='right')-1,0),len(plan['time'])-2)
        dt=plan['time'][j+1]-plan['time'][j];u=np.clip((t-plan['time'][j])/dt,0,1)
        q=(1-u)*plan['qpos'][j]+u*plan['qpos'][j+1];v=np.zeros(e.model.nv)
        v[:3]=(plan['qpos'][j+1,:3]-plan['qpos'][j,:3])/dt
        v[e.vadr]=(plan['qpos'][j+1,e.qadr]-plan['qpos'][j,e.qadr])/dt
        changed=z['qpos'][i];delta=changed[e.qadr[:6]]-q[e.qadr[:6]]
        v[e.vadr[:6]]+=(delta-previous)/.005;previous=delta
        velocity &= bool(np.allclose(v,z['qvel'][i],rtol=0,atol=1e-9))
        identity &= bool(np.array_equal(q[other],changed[other]))
        bounds &= bool(np.all(changed[e.qadr]>=e.model.jnt_range[e.joints,0]-1e-9) and np.all(changed[e.qadr]<=e.model.jnt_range[e.joints,1]+1e-9))
        scratch.qpos[:]=changed;nominal.qpos[:]=q
        mujoco.mj_kinematics(e.model,scratch);mujoco.mj_kinematics(e.model,nominal)
        body=e.model.body('left_ankle_roll_link').id
        expected=nominal.xpos[body]+[0,0,-rows[i]['depth_m']]
        position_error=max(position_error,float(np.linalg.norm(scratch.xpos[body]-expected)))
        orientation_error=max(orientation_error,float(np.max(abs(scratch.xmat[body]-nominal.xmat[body]))))
    confirmation=True
    transitions=[i for i,r in enumerate(rows) if r['mode']=='ramping' and (i==0 or rows[i-1]['mode']!='ramping')]
    for i in transitions:
        window=[r for r in rows[:i+1] if r['time_s']>=rows[i]['time_s']-.25-1e-8]
        confirmation &= len(window)>=51 and all(r['lf_valid'] for r in window)
    checks=dict(source_snapshot=all(hashlib.sha256((a.run/'source_snapshot'/rel).read_bytes()).hexdigest()==h for rel,h in manifest['sources'].items()),
        complete_reference_log=abs(z['time_s'][-1]+.005-summary['simulation_time_s'])<1e-7,
        finite=bool(np.isfinite(z['qpos']).all() and np.isfinite(z['qvel']).all()),
        native_joint_bounds=bounds,only_left_leg_changed=identity,velocity_consistent=velocity,
        bounded_search=all(0<=r['depth_m']<=.015+1e-9 for r in rows),
        kinematic_target=position_error<.002 and orientation_error<.01,
        force_caps=all(0<=r['lf_normal_cap_n']<=350 for r in rows),
        confirmation_history=bool(confirmation and len(transitions)==1),
        terminal_loaded=summary['landing']['mode']=='loaded')
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),audit_implementation='Same checks as v1; NPZ arrays materialized once to avoid repeated decompression; NumPy booleans converted for JSON',passed=bool(all(checks.values())),checks=checks,position_error_m=position_error,
        orientation_matrix_error=orientation_error,steps=counts,
        scope='Additional reference/state-machine audit. Confirmation checks logged privileged force/velocity validity; does not independently reconstruct contact forces or replace full-cycle audit.')
    (a.run/'landing_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
