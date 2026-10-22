"""Paired gain study; failed task stays failed when comparison integrity passes."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())

def main():
    suite=ROOT/'outputs/highrod_gain_sensitivity_v1';old=ROOT/'outputs/early_swing14_lift26_dynamic_v1/row01_z_+10mm';one=ROOT/'outputs/highrod_endpoint_adapt_dynamic_v1/row01_z_+10mm'
    reference=dict(np.load(old/'reference.npz'));adapt=dict(np.load(one/'reference.npz'));base=dict(np.load(old/'trajectory.npz'));m0=read(old/'manifest.json');rows=[]
    for gain,run in [(0.,old),(.75,suite/'gain_0.75'),(1.,one),(1.25,suite/'gain_1.25')]:
        z=dict(np.load(run/'trajectory.npz'));ref=dict(np.load(run/'reference.npz'));m=read(run/'manifest.json');configs=[dict(x['config']) for x in [m0,m]]
        for c in configs:c.pop('output_dir');c.pop('plan')
        prefix=base['time_s']<=99.66+1e-8;delta=float(np.max(abs(base['qpos'][prefix]-z['qpos'][prefix])))
        checks=dict(same_sources=m0['sources']==m['sources'],same_configuration=configs[0]==configs[1],same_geometry=read(old/'scene_variant.json')['geometry_sha256']==read(run/'scene_variant.json')['geometry_sha256'],same_schedule=bool(np.array_equal(ref['time'],reference['time']) and np.array_equal(ref['active'],reference['active']) and np.array_equal(ref['phase'],reference['phase'])),declared_gain=bool(np.allclose(ref['qpos'],reference['qpos']+gain*(adapt['qpos']-reference['qpos']),atol=1e-13,rtol=0)),identical_prefix=delta<1e-10)
        result=read(run/'summary.json');diagnosis=read(run/'endpoint_joint_diagnostic.json')
        rows.append(dict(gain=gain,comparison_integrity=all(checks.values()),checks=checks,prefix_qpos_difference=delta,task_passed=bool(result['passed'] and read(run/'independent_audit.json')['passed'] and read(run/'joint_limit_audit.json')['passed']),endpoint_peak_rad=abs(diagnosis['joints'][0]['peak_error_rad']),worst_joint=diagnosis['joints'][0]['joint'],joints=diagnosis['joints'],minimum_native_margin_rad=result['minimum_native_joint_margin_rad']))
    result=dict(integrity_passed=all(r['comparison_integrity'] for r in rows),results=rows,scope='Four discrete amplitudes, no continuous interval proof. Gain0and1 reuse earlier audited runs. Samephysicalprefix until99.66s. Failedclosure remainsfailed, nativebounds remainunchanged.')
    (suite/'paired_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({**result,'results':[{k:v for k,v in r.items() if k not in ['joints','checks']} for r in rows]},indent=2))
    if not result['integrity_passed']:raise SystemExit(1)

if __name__=='__main__':main()
