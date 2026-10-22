"""Blend bounded offline endpoint correction during existing posture closure."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.hand_transfer import smooth
from tools.plan_clearance_cycle import audit_plan

def main():
    parent=ROOT/'outputs/early_swing14_lift26_plan_v1/plan.npz'
    probe=ROOT/'outputs/highrod_endpoint_minimax_v1'
    candidate=json.loads((probe/'summary.json').read_text());assert candidate['candidate_verified']
    endpoint=probe/'candidate.npz'
    out=ROOT/'outputs/highrod_endpoint_minimax_plan_v1';out.mkdir(exist_ok=False)
    plan=dict(np.load(parent));target=dict(np.load(endpoint))['qpos'];delta=target-plan['qpos'][-1]
    start=float(plan['time'][np.flatnonzero(plan['phase']=='posture_closure')[0]-1])
    end=float(plan['time'][np.flatnonzero(plan['phase']=='closed_hold')[0]-1])
    w=smooth((plan['time']-start)/(end-start));plan['qpos']=plan['qpos']+w[:,None]*delta
    np.savez_compressed(out/'plan.npz',**plan)
    e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');audit=audit_plan(e,plan)
    audit['scope']='Nominal-scene dense collision/native-position check; errors array inherited from parent, not a fresh IK certificate for blended targets. Perturbed rod contact dynamics requires separate execution.'
    audit['parent_ik_residual']=audit.pop('ik_residual')
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    manifest=dict(parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),endpoint=str(endpoint),endpoint_sha256=hashlib.sha256(endpoint.read_bytes()).hexdigest(),selection='Local minimax jointdeviation with exact limbposes',blend_window_s=[start,end],selected=candidate,plan_sha256=hashlib.sha256((out/'plan.npz').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Known+10mm rod geometry used offline. Privileged geometry-adapted terminal target, not perception policy or universal periodic reference. Existing0.04rad jointclosure and dynamic thresholds unchanged. No claim of repeatability acrosscycles.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));e.close()
    print(json.dumps(dict(audit={k:v for k,v in audit.items() if k!='failures'},manifest=manifest),indent=2))
    if not audit['passed']:raise SystemExit(1)

if __name__=='__main__':main()
