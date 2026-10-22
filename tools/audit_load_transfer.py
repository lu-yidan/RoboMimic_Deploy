"""Audit successful and failed load-transfer evaluations without hiding failures."""
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
    manifest=json.loads((a.run/'manifest.json').read_text())
    evaluation=(a.run/'summary.json').exists()
    summary=json.loads((a.run/'summary.json').read_text()) if evaluation else {'baseline':json.loads((a.run/'results.json').read_text())}
    sources_ok=all(sha(a.run/'source_snapshot'/rel)==value for rel,value in manifest['sources'].items())
    prefixes_ok=all(sha(Path(manifest['suite'])/name/'trajectory.npz')==value for name,value in manifest['prefixes'].items())
    results=[]
    for label in (('zero','before','after') if evaluation else ('baseline',)):
        for row in summary[label]:
            name=row['case'];e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
            apply_rung_offsets(e,row['offsets']);m=e.model;d=e.data
            z=np.load(a.run/(f'{label}_{name}.npz' if evaluation else name+'.npz'))
            with np.load(Path(manifest['suite'])/name/'trajectory.npz') as prefix:
                i=int(np.argmin(abs(prefix['time_s']-49.)))
                initial_ok=np.array_equal(z['qpos'][0],prefix['qpos'][i]) and np.array_equal(z['qvel'][0],prefix['qvel'][i])
            expected={m.body(n).id:r for n,r in [('right_ankle_roll_link','rung_01'),('hook','rung_05'),('right_hook','rung_05')]}
            left=m.body('left_ankle_roll_link').id
            penetration=0.;terminal_ok=True;terminal_samples=0
            for t,q in zip(z['time_s'],z['qpos']):
                d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d);touched=set()
                for c in d.contact:
                    ba,bb=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
                    ia,ib=ba in e.robot_bodies,bb in e.robot_bodies
                    if not (ia or ib):continue
                    body=ba if ia else bb
                    if ia==ib or body not in e.contact_groups:penetration=max(penetration,float(-c.dist))
                    if t>61.31 and ia!=ib:
                        external=m.geom(c.geom2 if ia else c.geom1).name
                        if body==left and c.dist<0:terminal_ok=False
                        if body in expected:
                            if external==expected[body]:touched.add(body)
                            else:terminal_ok=False
                if t>61.31:
                    terminal_samples+=1;terminal_ok &= touched==set(expected)
            dt=np.diff(z['time_s'])
            temporal=bool(np.all(dt>=-1e-10) and np.all(dt<=.0200001) and
                          (np.all(dt>0) or (not row['success'] and np.all(dt[:-1]>0) and abs(dt[-1])<1e-10)))
            complete=abs(z['time_s'][-1]-61.56)<1e-7
            success_geometry=bool(complete and terminal_ok and terminal_samples>=12)
            checks=dict(frozen_sources=sources_ok,unchanged_prefixes=prefixes_ok,recorded_initial_state=bool(initial_ok),
                timestamps=temporal,finite_states=bool(np.isfinite(z['qpos']).all() and np.isfinite(z['qvel']).all()),
                no_unexpected_penetration=penetration<1e-5,
                recorded_torque_bounds=bool(np.all(z['torques']>=e.limits[:,0]-1e-5) and np.all(z['torques']<=e.limits[:,1]+1e-5)),
                action_contract=bool(z['actions'].shape==(len(z['time_s'])-1,6) and np.isfinite(z['actions']).all() and np.max(abs(z['actions']))<=1.+1e-6) if evaluation else 'zero residual' in manifest['contract'],
                observation_contract=bool(z['observations'].shape==(len(z['time_s'])-1,76) and np.isfinite(z['observations']).all()),
                reported_success_supported=not row['success'] or success_geometry)
            results.append(dict(label=label,case=name,integrity_passed=all(checks.values()),checks=checks,
                runtime_success=row['success'],terminal_geometry_passed=success_geometry,
                unexpected_penetration_m=penetration,time_s=float(z['time_s'][-1])))
            e.close()
    result=dict(integrity_passed=all(r['integrity_passed'] for r in results),results=results,
        scope='50 Hz sampled geometry/torques; successful terminal contact geometry checked. Integrity passing does not turn a failed task into success. Runtime confirmation supplies force evidence.',
        action_audit='Recorded policy actions checked' if evaluation else 'Zero-residual diagnostic source contract only; no action samples or policy-action audit claim')
    (a.run/'independent_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
