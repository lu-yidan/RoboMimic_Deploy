"""Independent sampled-state geometry and torque audit for foot PPO evaluations."""
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
from training.climb_robustness import apply_rung_offsets


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text());summary=json.loads((a.run/'summary.json').read_text())
    source_ok=all(sha(a.run/'source_snapshot'/rel)==value for rel,value in manifest['sources'].items())
    prefix_ok=all(sha(Path(manifest['suite'])/name/'trajectory.npz')==value for name,value in manifest['prefixes'].items())
    results=[]
    for label in ('zero','before','after'):
        for row in summary[label]:
            name=row['case'];e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
            apply_rung_offsets(e,row['offsets']);m=e.model;d=e.data
            z=np.load(a.run/f'{label}_{name}.npz')
            maximum=0.;swing=0.;terminal_ok=True;terminal_samples=0
            expected={m.body(n).id:r for n,r in [('left_ankle_roll_link','rung_00'),('right_ankle_roll_link','rung_01'),('hook','rung_05'),('right_hook','rung_05')]}
            for t,q in zip(z['time_s'],z['qpos']):
                d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
                touched=set()
                for c in d.contact:
                    ba,bb=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
                    ia,ib=ba in e.robot_bodies,bb in e.robot_bodies
                    if not (ia or ib):continue
                    body=ba if ia else bb
                    intended=ia!=ib and body in e.contact_groups
                    if not intended:maximum=max(maximum,float(-c.dist))
                    if ia!=ib and body==m.body('right_ankle_roll_link').id and 37.005<=t<45.56-1e-7:
                        swing=max(swing,float(-c.dist))
                    if t>49.31 and intended:
                        external=m.geom(c.geom2 if ia else c.geom1).name
                        if external==expected[body]:touched.add(body)
                        else:terminal_ok=False
                if t>49.31:
                    terminal_samples+=1;terminal_ok &= touched==set(expected)
            checks=dict(frozen_sources=source_ok,unchanged_prefixes=prefix_ok,
                        full_duration=bool(abs(z['time_s'][0]-34.5)<1e-8 and abs(z['time_s'][-1]-49.56)<1e-7),
                        finite_states=bool(np.isfinite(z['qpos']).all() and np.isfinite(z['qvel']).all()),
                        no_unexpected_penetration=maximum<1e-5,no_swing_penetration=swing<1e-5,
                        terminal_contact_geometry=bool(terminal_ok and terminal_samples>=12),
                        recorded_torque_bounds=bool(np.all(z['torques']>=e.limits[:,0]-1e-5) and np.all(z['torques']<=e.limits[:,1]+1e-5)),
                        runtime_success=bool(row['success']))
            results.append(dict(label=label,case=name,passed=all(checks.values()),checks=checks,
                                unexpected_penetration_m=maximum,swing_penetration_m=swing))
            e.close()
    result=dict(passed=all(r['passed'] for r in results),results=results,
                scope='50 Hz recorded state geometry and torque audit; force success uses 200 Hz runtime guard. No independent force reconstruction or held-out learning claim.')
    (a.run/'independent_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
