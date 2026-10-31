"""Separate policy-action clipping from native actuator-command saturation."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.compare_audited_closure_candidates import load_run,sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 verified=load_run(a.experiment);m=json.loads((a.experiment/'manifest.json').read_text());env=FullCycleResidualEnv(plan=m['reference'],cases=m['evaluation_cases']);rows=[]
 try:
  for case in m['evaluation_cases']:
   env.reset(options={'case':case});e=env.env
   with np.load(a.experiment/f'after_{case}.npz') as z:
    # Saved torque belongs to the transition ending at this state time.
    torques=z['torques'][z['time_s'][1:]>111.66]
   joints=[]
   for j in range(29):
    lo,hi=e.limits[j];values=torques[:,j]
    joints.append(dict(joint=e.model.joint(int(e.joints[j])).name,command_range_nm=[float(values.min()),float(values.max())],native_range_nm=[float(lo),float(hi)],
     positive_saturation_fraction=float(np.mean(values>=hi-1e-7)),negative_saturation_fraction=float(np.mean(values<=lo+1e-7)),minimum_command_margin_nm=float(np.minimum(values-lo,hi-values).min())))
   rows.append(dict(case=case,success=verified['rows'][case]['success'],samples=len(torques),joints=joints))
 finally:env.close()
 a.output.mkdir(exist_ok=False);result=dict(rows=rows,audit_sha256=verified['audit_sha256'],source_sha256=sha(__file__),
  scope='Saved final 200 Hz controller command sampled at 50 Hz transition endpoints. Not complete physics-step torque peaks or achieved torque. Actor clipping and actuator clipping are different constraints.')
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));(a.output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps([{**{k:r[k] for k in ('case','success')},'left_leg':[j for j in r['joints'] if j['joint'] in ('left_knee_joint','left_ankle_pitch_joint')]} for r in rows]))

if __name__=='__main__':main()
