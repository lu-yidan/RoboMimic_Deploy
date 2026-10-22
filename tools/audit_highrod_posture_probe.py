"""Check saved IK candidate geometry independently of the optimizer objective."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.repeated_rung_errors import apply_rung_offsets
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args();p=args.experiment
    protocol=json.loads((p/'protocol.json').read_text());summary=json.loads((p/'summary.json').read_text())
    assert sha(p/'probe_source.py')==summary['source_sha256'] and sha(p/'protocol.json')==summary['protocol_sha256']
    results=[]
    for label,row in zip(protocol['cases'],summary['results']):
        source=Path('outputs/highrod_height_mismatch_v1')/label;m=json.loads((source/'manifest.json').read_text())
        assert sha(source/'trajectory.npz')==row['source_trajectory_sha256']
        with np.load(source/'trajectory.npz') as z:original=z['qpos'][-1].copy()
        e=WallSupportEnv('outputs/wall_support_bundle_pd15_v2');e.reset();apply_rung_offsets(e,m['offsets']);model=e.model
        d=mujoco.MjData(model);bodies=[model.body(n).id for n in ('hook','right_hook','left_ankle_roll_link','right_ankle_roll_link')]
        d.qpos[:]=original;mujoco.mj_kinematics(model,d);target=d.xpos[bodies].copy();rotation=d.xmat[bodies].copy()
        ankle=model.joint('right_ankle_pitch_joint').id
        for seed in range(len(row['candidates'])):
            file=p/f'{label}_seed{seed}.npz'
            with np.load(file) as z:q=z['candidate_qpos'].copy();same_original=np.array_equal(z['original_qpos'],original)
            d.qpos[:]=q;mujoco.mj_kinematics(model,d);mujoco.mj_collision(model,d)
            position=float(np.linalg.norm(d.xpos[bodies]-target,axis=1).max())
            angles=[]
            for aa,bb in zip(d.xmat[bodies],rotation):angles.append(float(np.arccos(np.clip((np.trace(aa.reshape(3,3)@bb.reshape(3,3).T)-1)/2,-1.,1.))))
            margin=float(np.minimum(q[e.qadr]-model.jnt_range[e.joints,0],model.jnt_range[e.joints,1]-q[e.qadr]).min())
            penetration=0.
            for c in d.contact:
                a,b=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2]);ia,ib=a in e.robot_bodies,b in e.robot_bodies
                if not (ia or ib):continue
                expected=ia!=ib and (a if ia else b) in e.contact_groups
                if not expected:penetration=max(penetration,float(-c.dist))
            checks=dict(original_state=same_original,base_orientation=np.array_equal(q[3:7],original[3:7]),
                base_translation=np.linalg.norm(q[:3]-original[:3])<=.005+1e-12,
                joint_deviation=abs(q[e.qadr]-original[e.qadr]).max()<=.015+1e-12,
                native_bounds=margin>=-1e-7,ankle_margin=q[model.jnt_qposadr[ankle]]-model.jnt_range[ankle,0]>=.005-1e-9,
                four_positions=position<1e-5,four_orientations=max(angles)<1e-4,collision=penetration<1e-5,
                waist_lock=not protocol.get('waist_locked',False) or np.array_equal(q[e.qadr[12:15]],original[e.qadr[12:15]]))
            results.append(dict(case=row['case'],seed=seed,passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
                position_error_m=position,rotation_error_rad=max(angles),minimum_native_margin_rad=margin,candidate_sha256=sha(file)))
        e.close()
    result=dict(passed=all(r['passed'] for r in results),results=results,summary_sha256=sha(p/'summary.json'),source_sha256=sha(__file__),scope='Independent static geometry verification only; no control command, velocity, force balance or successful dynamic transition is established.')
    (p/'independent_audit.json').write_text(json.dumps(result,indent=2));(p/'audit_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(passed=result['passed'],candidates=len(results))))
    if not result['passed']:raise AssertionError(result)


if __name__=='__main__':main()
