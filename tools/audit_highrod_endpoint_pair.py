"""Verify endpoint-only reference correction against unchanged highrod control."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())

def main():
    old=ROOT/'outputs/early_swing14_lift26_dynamic_v1/row01_z_+10mm';new=ROOT/'outputs/highrod_endpoint_adapt_dynamic_v1/row01_z_+10mm'
    plan=read(ROOT/'outputs/highrod_endpoint_adapt_plan_v1/manifest.json');start,end=plan['blend_window_s']
    manifests=[read(run/'manifest.json') for run in [old,new]];configs=[dict(m['config']) for m in manifests]
    for c in configs:c.pop('output_dir');c.pop('plan')
    refs=[dict(np.load(run/'reference.npz')) for run in [old,new]];states=[dict(np.load(run/'trajectory.npz')) for run in [old,new]]
    early=refs[0]['time']<=start;prefix=states[0]['time_s']<=start+1e-8
    delta=float(np.max(abs(states[0]['qpos'][prefix]-states[1]['qpos'][prefix])))
    audits={name:read(new/name)['passed'] for name in ['summary.json','independent_audit.json','joint_limit_audit.json','contact_identity_audit.json']}
    checks=dict(same_control_sources=manifests[0]['sources']==manifests[1]['sources'],same_configuration=configs[0]==configs[1],same_geometry=read(old/'scene_variant.json')['geometry_sha256']==read(new/'scene_variant.json')['geometry_sha256'],same_times=bool(np.array_equal(refs[0]['time'],refs[1]['time'])),same_active=bool(np.array_equal(refs[0]['active'],refs[1]['active'])),same_phases=bool(np.array_equal(refs[0]['phase'],refs[1]['phase'])),unchanged_reference_before_closure=bool(np.array_equal(refs[0]['qpos'][early],refs[1]['qpos'][early])),same_start_pose=bool(np.array_equal(refs[0]['qpos'][0],refs[1]['qpos'][0])),identical_physical_prefix=delta<1e-10,all_new_audits=all(audits.values()))
    result=dict(passed=all(checks.values()),checks=checks,physical_prefix_difference=delta,blend_window_s=[start,end],before_passed=read(old/'summary.json')['passed'],after_passed=read(new/'summary.json')['passed'],audits=audits,scope='Same highrod scene, timing, control and acceptance. Only closure reference changes from99.66s. Offline true10mm offset used; no uncertainty robustness, repeated-cycle, policy or hardware claim.')
    (new/'endpoint_pair_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
