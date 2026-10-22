"""Independent checks on saved cycle states, contact logs and frozen sources."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_robustness import apply_rung_offsets


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit_endpoints(e,z,ref):
    ends=np.asarray(ref.get('cycle_ends_s',[ref['time'][-1]]));result=[]
    d=mujoco.MjData(e.model)
    bodies=[e.model.body(n).id for n in ['hook','right_hook','left_ankle_roll_link','right_ankle_roll_link']]
    for k,end in enumerate(ends):
        j=int(np.argmin(abs(ref['time']-end)))
        d.qpos[:]=ref['qpos'][j];mujoco.mj_kinematics(e.model,d);desired=d.xpos[bodies].copy()
        ids=np.flatnonzero((z['time_s']>end-2.)&(z['time_s']<=end+1e-7));pose=0.;tilt=0.;closure=0.
        for i in ids:
            d.qpos[:]=z['qpos'][i];mujoco.mj_kinematics(e.model,d)
            pose=max(pose,float(np.linalg.norm(d.xpos[bodies]-desired,axis=1).max()))
            tilt=max(tilt,float(np.arccos(np.clip(d.xmat[e.pelvis].reshape(3,3)[2,2],-1,1))))
            closure=max(closure,float(abs(z['qpos'][i,e.qadr]-ref['qpos'][0,e.qadr]).max()))
        linear=float(np.linalg.norm(z['qvel'][ids,:3],axis=1).max()) if len(ids) else None
        angular=float(np.linalg.norm(z['qvel'][ids,3:6],axis=1).max()) if len(ids) else None
        joint=float(abs(z['qvel'][ids,6:]).max()) if len(ids) else None
        passed=len(ids)>=99 and pose<.01 and tilt<np.deg2rad(3) and closure<.04 and linear<.01 and angular<.1 and joint<.1
        result.append(dict(cycle=k+1,passed=bool(passed),sample_count=len(ids),effector_error_m=pose if len(ids) else None,
                           tilt_deg=float(np.rad2deg(tilt)) if len(ids) else None,joint_closure_rad=closure if len(ids) else None,
                           linear_speed_m_s=linear,angular_speed_rad_s=angular,joint_speed_rad_s=joint))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text());summary=json.loads((a.run/'summary.json').read_text())
    e=WallSupportEnv(manifest['bundle']);e.reset()
    variant_path=a.run/'scene_variant.json'
    variant=json.loads(variant_path.read_text()) if variant_path.exists() else None
    applied=apply_rung_offsets(e,variant['offsets'] if variant else {})
    if variant and applied!=variant:raise ValueError('Scene variant reconstruction mismatch')
    z=dict(np.load(a.run/'trajectory.npz'));ref=dict(np.load(a.run/'reference.npz'))
    initial=ref['qpos'][0];final=ref['qpos'][-1];tail=np.flatnonzero(z['time_s']>ref['time'][-1]-2.)
    bodies=[e.model.body(n).id for n in ['hook','right_hook','left_ankle_roll_link','right_ankle_roll_link']]
    e.data.qpos[:]=final;mujoco.mj_forward(e.model,e.data);desired=e.data.xpos[bodies].copy()
    pose_error=0.;tilt=0.;joint_error=0.;unexpected_penetration=0.;first_unexpected=None
    for i,q in enumerate(z['qpos']):
        e.data.qpos[:]=q;mujoco.mj_kinematics(e.model,e.data);mujoco.mj_collision(e.model,e.data)
        if i in tail:
            pose_error=max(pose_error,float(np.max(np.linalg.norm(e.data.xpos[bodies]-desired,axis=1))))
            tilt=max(tilt,float(np.arccos(np.clip(e.data.xmat[e.pelvis].reshape(3,3)[2,2],-1,1))))
            joint_error=max(joint_error,float(np.max(abs(q[e.qadr]-initial[e.qadr]))))
        for c in e.data.contact:
            ba,bb=int(e.model.geom_bodyid[c.geom1]),int(e.model.geom_bodyid[c.geom2])
            ina,inb=ba in e.robot_bodies,bb in e.robot_bodies
            if not (ina or inb):continue
            expected=ina!=inb and (ba if ina else bb) in e.contact_groups
            if not expected and c.dist<0:
                unexpected_penetration=max(unexpected_penetration,float(-c.dist))
                if first_unexpected is None:
                    first_unexpected=dict(time_s=float(z['time_s'][i]),bodies=[e.model.body(ba).name,e.model.body(bb).name])
    source_ok=all(sha(a.run/'source_snapshot'/s)==h for s,h in manifest['sources'].items())
    linear_speed=float(np.max(np.linalg.norm(z['qvel'][tail,:3],axis=1))) if len(tail) else None
    angular_speed=float(np.max(np.linalg.norm(z['qvel'][tail,3:6],axis=1))) if len(tail) else None
    joint_speed=float(np.max(abs(z['qvel'][tail,6:]))) if len(tail) else None
    checks=dict(run_passed=bool(summary['passed']),frozen_sources_match=source_ok,
                frozen_reference_matches=sha(a.run/'reference.npz')==manifest['plan_sha256'],
                uniform_50hz_recording=bool(np.allclose(np.diff(z['time_s']),.02,atol=1e-8)),
                finite_states=bool(np.isfinite(z['qpos']).all() and np.isfinite(z['qvel']).all()),
                final_effector_alignment=bool(len(tail)) and pose_error<.01,
                final_upright=bool(len(tail)) and tilt<np.deg2rad(3),
                final_joint_closure=bool(len(tail)) and joint_error<.04,
                final_settled_velocity=bool(len(tail)) and linear_speed<.01 and angular_speed<.1 and joint_speed<.1,
                sampled_unexpected_penetration=unexpected_penetration<1e-5,
                recorded_torque_bounds=bool(np.all(z['torques']>=e.limits[:,0]-1e-5) and np.all(z['torques']<=e.limits[:,1]+1e-5)))
    endpoints=audit_endpoints(e,z,ref)
    checks['every_cycle_endpoint']=all(r['passed'] for r in endpoints)
    checks={k:bool(v) for k,v in checks.items()}
    utilization=abs(z['torques'])/np.maximum(abs(e.limits[:,0]),abs(e.limits[:,1]))
    result=dict(audit_source_sha256=sha(__file__),passed=all(checks.values()),checks=checks,cycle_endpoints=endpoints,final_effector_error_peak_m=pose_error,
                final_tilt_peak_deg=float(np.rad2deg(tilt)),final_joint_closure_peak_rad=joint_error,
                sampled_unexpected_penetration_m=unexpected_penetration,first_unexpected=first_unexpected,
                final_linear_speed_peak_m_s=linear_speed,final_angular_speed_peak_rad_s=angular_speed,
                final_joint_speed_peak_rad_s=joint_speed,final_sample_count=int(len(tail)),
                torque_peak_ratio_by_joint_50hz={e.model.joint(j).name:float(r) for j,r in zip(e.joints,utilization.max(axis=0))},
                torque_near_limit_fraction_50hz=float(np.mean(utilization.max(axis=1)>.99)),
                trajectory_sha256=sha(a.run/'trajectory.npz'),bundle_manifest_sha256=sha(e.bundle/'manifest.json'),
                note='Offline geometry at saved 50 Hz states. Force peaks remain the live 2000 Hz measurements in summary.json.')
    (a.run/'independent_audit.json').write_text(json.dumps(result,indent=2))
    (a.run/'independent_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2));e.close()


if __name__=='__main__':main()
