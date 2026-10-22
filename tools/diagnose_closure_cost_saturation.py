"""Prospective reward-signal diagnostic; does not alter the frozen training run."""
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.closure_margin_reward import closure_cost
from tools.continue_vector_full_cycle_training import sha


def main():
    out=Path('outputs/closure_cost_saturation_v1');out.mkdir(exist_ok=False);rows=[]
    root=Path('outputs/support_retarget_calibration_v1')
    for result in json.loads((root/'results.json').read_text()):
        p=root/result['label'];meta=json.loads((p/'manifest.json').read_text());audit=json.loads((p/'independent_audit.json').read_text());assert audit['integrity_passed'] and audit['trajectory_sha256']==sha(p/'trajectory.npz')
        env=FullCycleResidualEnv(plan=meta['reference']);e=env.env
        try:
            with np.load(p/'trajectory.npz') as z:
                q=z['qpos'];time=z['time_s'];closure=abs(q[:,e.qadr]-q[0,e.qadr]).max(axis=1);rise=q[:,2]-q[0,2]
                intervals={}
                for name,start in [('full_weight',102.66),('final_two_seconds',111.66)]:
                    ids=np.flatnonzero(time>start);c=closure[ids];h=rise[ids];cost=np.array([closure_cost(float(a),float(b),1.) for a,b in zip(c,h)])
                    improved_joint=np.array([closure_cost(max(0.,float(a)-.001),float(b),1.) for a,b in zip(c,h)])
                    improved_height=np.array([closure_cost(float(a),float(b)-np.sign(b-.25)*min(.00025,abs(float(b)-.25)),1.) for a,b in zip(c,h)])
                    intervals[name]=dict(samples=len(ids),joint_term_saturated_fraction=float(np.mean(c>=.04)),height_term_saturated_fraction=float(np.mean(abs(h-.25)>=.01)),both_saturated_fraction=float(np.mean((c>=.04)&(abs(h-.25)>=.01))),mean_cost=float(cost.mean()),mean_cost_reduction_if_closure_improves_1mrad=float((cost-improved_joint).mean()),mean_cost_reduction_if_height_improves_025mm=float((cost-improved_height).mean()))
                rows.append(dict(case=meta['case'],accepted=audit['independently_accepted'],intervals=intervals,trajectory_sha256=sha(p/'trajectory.npz')))
        finally:env.close()
    result=dict(rows=rows,source_sha256=sha(__file__),reward_source_sha256=sha('training/closure_margin_reward.py'),scope='Scalar perturbations of recorded metrics, not physically realizable action perturbations or gradient-through-dynamics. Quantifies flat regions of the bounded reward. Current matched training remains unchanged; not evidence of learning performance.')
    (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))


if __name__=='__main__':main()
