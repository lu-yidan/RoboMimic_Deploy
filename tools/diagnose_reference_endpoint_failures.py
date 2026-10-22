"""Report numeric terminal criteria without relaxing or relabelling acceptance."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args();rows=[]
    for record in json.loads((args.experiment/'results.json').read_text()):
        if record['reason']!='cycle_endpoint_failure':continue
        p=args.experiment/record['label'];m=json.loads((p/'manifest.json').read_text());a=json.loads((p/'independent_audit.json').read_text())
        assert a['integrity_passed'] and a['trajectory_sha256']==sha(p/'trajectory.npz')
        env=FullCycleResidualEnv(plan=m['reference']);e=env.env
        try:
            with np.load(p/'trajectory.npz') as z:
                q=z['qpos'];t=z['time_s'];obs=z['observations'];tail=(t>env.horizon-2)&(t<=env.horizon+1e-7)
                closure=abs(q[tail][:,e.qadr]-q[0,e.qadr]);per_joint=closure.max(axis=0);k=int(per_joint.argmax())
                rise=q[tail,2]-q[0,2];force_mask=tail[:-1];forces=obs[force_mask,106:110]*e.robot_weight_n
                rows.append(dict(case=m['case'],max_joint_closure_rad=float(per_joint[k]),closure_joint=e.model.joint(e.joints[k]).name,
                    closure_passed=bool(per_joint[k]<.04),root_rise_min_m=float(rise.min()),root_rise_max_m=float(rise.max()),
                    root_rise_passed=bool(np.all(abs(rise-.25)<.01)),minimum_vertical_support_n=forces.min(axis=0).tolist(),
                    force_sample_count=int(force_mask.sum()),minimum_vertical_support_passed=bool(np.min(forces)>10.),
                    strict_endpoint=a['endpoints'],trajectory_sha256=sha(p/'trajectory.npz'),audit_sha256=sha(p/'independent_audit.json')))
        finally:env.close()
    result=dict(rows=rows,source_sha256=sha(__file__),scope='Numeric saved-tail diagnostics. Contact identity remains runtime evidence and is not reconstructed here. Original strict acceptance unchanged.')
    (args.experiment/'endpoint_diagnostic.json').write_text(json.dumps(result,indent=2));(args.experiment/'endpoint_diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
