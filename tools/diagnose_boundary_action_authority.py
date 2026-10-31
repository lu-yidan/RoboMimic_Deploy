"""Describe recorded action saturation without claiming controller infeasibility."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.compare_audited_closure_candidates import load_run, sha
from training.full_cycle_residual_env import FullCycleResidualEnv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); parent=load_run(a.parent)
    m=json.loads((a.parent/'manifest.json').read_text())
    env=FullCycleResidualEnv(plan=m['reference'],cases=m['evaluation_cases'])
    names=[env.env.model.joint(int(env.env.joints[i])).name for i in env.indices]
    env.close();rows=[]
    for case in ['boundary_04','boundary_05','boundary_06','boundary_07']:
        path=a.parent/f'after_{case}.npz'
        with np.load(path) as z:
            windows={}
            for name,start in [('landing',70.2),('last_two_seconds',111.66)]:
                actions=z['actions'][z['time_s'][:-1]>=start]
                windows[name]=[dict(joint=joint,minimum=float(actions[:,i].min()),maximum=float(actions[:,i].max()),
                    mean=float(actions[:,i].mean()),positive_saturation_fraction=float(np.mean(actions[:,i]>=1.)),
                    negative_saturation_fraction=float(np.mean(actions[:,i]<=-1.))) for i,joint in enumerate(names)]
        rows.append(dict(case=case,trajectory_sha256=sha(path),windows=windows))
    report=dict(parent=str(a.parent.resolve()),parent_audit_sha256=parent['audit_sha256'],rows=rows,
        source_sha256=sha(__file__),scope='Recorded normalized policy actions only. Saturation does not prove global infeasibility, physical torque saturation, or a need to expand limits.')
    a.output.mkdir(exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps([dict(case=r['case'],last_two_seconds=[x for x in r['windows']['last_two_seconds'] if x['joint'] in ('left_knee_joint','left_ankle_pitch_joint')]) for r in rows]),flush=True)


if __name__=='__main__':main()
