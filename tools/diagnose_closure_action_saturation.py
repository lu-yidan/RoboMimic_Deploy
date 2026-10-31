"""Locate worst joint error and actual action saturation in audited endpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.compare_audited_closure_candidates import load_run,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verified=load_run(args.experiment);manifest=json.loads((args.experiment/'manifest.json').read_text())
    env=FullCycleResidualEnv(plan=manifest['reference'],cases=manifest['evaluation_cases']);rows=[]
    try:
        for case in manifest['evaluation_cases']:
            env.reset(options={'case':case});e=env.env
            with np.load(args.experiment/f'after_{case}.npz') as trace:
                mask=trace['time_s']>111.66;action_mask=trace['time_s'][:-1]>111.66
                delta=trace['qpos'][mask][:,e.qadr]-e.initial[e.qadr]
                sample,joint=np.unravel_index(abs(delta).argmax(),delta.shape)
                assert abs(abs(delta[sample,joint])*1000-verified['rows'][case]['closure_mrad'])<1e-8
                mapped=np.flatnonzero(env.indices==joint)
                action=trace['actions'][action_mask,int(mapped[0])] if len(mapped) else None
                row=dict(case=case,task_success=verified['rows'][case]['success'],joint=e.model.joint(int(e.joints[joint])).name,
                    signed_deviation_mrad=float(delta[sample,joint]*1000),time_s=float(trace['time_s'][mask][sample]),
                    actor_action_index=int(mapped[0]) if len(mapped) else None,
                    action_range=[float(action.min()),float(action.max())] if action is not None else None,
                    positive_saturation_fraction=float(np.mean(action>=1.-1e-7)) if action is not None else None,
                    negative_saturation_fraction=float(np.mean(action<=-1.+1e-7)) if action is not None else None)
                rows.append(row)
    finally:env.close()
    args.output.mkdir(exist_ok=False)
    result=dict(rows=rows,audit_sha256=verified['audit_sha256'],checkpoint_sha256=verified['checkpoint_sha256'],source_sha256=sha(__file__),
        scope='Recorded action clipping and joint error association. Not a causal attribution, whole-action-space infeasibility proof, or justification for changing task/actuator limits.')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2));(args.output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
