"""Inspect saved terminal contact geometry; never infer runtime force from replay."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args()
    p=args.experiment;m=json.loads((p/'manifest.json').read_text());s=json.loads((p/'summary.json').read_text())
    a=json.loads((p/'independent_training_audit.json').read_text())
    if not a['passed'] or sha(p/'summary.json')!=a['experiment_summary_sha256']:raise ValueError('Audited evidence required')
    case='foot_z_plus10';r=next(x for x in s['after'] if x['case']==case);path=p/f'after_{case}.npz'
    if sha(path)!=r['trajectory_sha256']:raise ValueError('Trajectory changed')
    env=FullCycleResidualEnv(plan=m['reference'],cases={case:m['evaluation_cases'][case]})
    env.reset(options={'case':case});model=env.env.model;data=mujoco.MjData(model)
    with np.load(path) as z:
        data.qpos[:]=z['qpos'][-1];t=float(z['time_s'][-1]);phase=env.reference(t)[3]
    mujoco.mj_kinematics(model,data);mujoco.mj_collision(model,data)
    contacts=[]
    for c in data.contact:
        ids=[int(c.geom1),int(c.geom2)]
        bodies=[int(model.geom_bodyid[g]) for g in ids]
        if not any(b in env.env.robot_bodies for b in bodies):continue
        contacts.append(dict(geoms=[model.geom(g).name for g in ids],
            bodies=[model.body(b).name for b in bodies],distance_m=float(c.dist),position_world=c.pos.tolist()))
    result=dict(case=case,time_s=t,phase=phase,runtime_reason=r['reason'],runtime_peaks=r['peaks'],
        reconstructed_contact_geometry=contacts,trajectory_sha256=sha(path),audit_sha256=sha(p/'independent_training_audit.json'),
        source_sha256=sha(__file__),scope='Collision geometry at saved terminal pose only. Contact forces are not reconstructed; runtime peaks are original logs. No new dynamics or causal control ablation.')
    (p/'terminal_collision_geometry.json').write_text(json.dumps(result,indent=2))
    (p/'terminal_collision_geometry_source.py').write_bytes(Path(__file__).read_bytes());env.close()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
