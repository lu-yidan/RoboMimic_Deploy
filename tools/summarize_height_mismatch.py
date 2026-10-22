"""Summarize fixed-reference actual height errors and a common geometry metric."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args();p=args.experiment
    protocol=json.loads((p/'protocol.json').read_text());comparison=json.loads((p/'independent_comparison.json').read_text());rows=[]
    reference_hashes=set();source_sets=[]
    for r in comparison['results']:
        run=p/r['label'];m=json.loads((run/'manifest.json').read_text());s=json.loads((run/'summary.json').read_text());a=json.loads((run/'independent_audit.json').read_text())
        assert r['integrity_passed'] and sha(run/'independent_audit.json')==r['audit_sha256'] and a['summary_sha256']==sha(run/'summary.json') and a['trajectory_sha256']==sha(run/'trajectory.npz')
        assert m['normal_bias_y_deg']==-3. and m['memory_enabled'] and m['protocol_sha256']==sha(p/'protocol.json')
        reference_hashes.add(m['reference_sha256']);source_sets.append(m['sources'])
        height=1000*m['offsets']['rung_01'][2]
        env=FullCycleResidualEnv(plan=m['reference'],cases={m['case']:m['offsets']});env.reset(options={'case':m['case']});e=env.env;d=mujoco.MjData(e.model)
        nominal_path=Path('outputs/early_swing14_lift26_plan_v1/plan.npz')
        with np.load(nominal_path) as nominal:d.qpos[:]=nominal['qpos'][-1]
        mujoco.mj_kinematics(e.model,d)
        bodies=[e.model.body(n).id for n in ('hook','right_hook','left_ankle_roll_link','right_ankle_roll_link')]
        desired=d.xpos[bodies].copy();desired[2:,2]+=height/1000.
        error=np.zeros(4)
        with np.load(run/'trajectory.npz') as z:
            ids=np.flatnonzero((z['time_s']>env.horizon-2)&(z['time_s']<=env.horizon+1e-7))
            for i in ids:
                d.qpos[:]=z['qpos'][i];mujoco.mj_kinematics(e.model,d);error=np.maximum(error,np.linalg.norm(d.xpos[bodies]-desired,axis=1))
        rows.append(dict(true_height_mm=height,error_true_minus_assumed_mm=height-protocol['assumed_height_mm'],accepted=r['accepted'],reason=s['reason'],time_s=s['time_s'],
            runtime_peaks=s['peaks'],minimum_native_joint_margin_rad=s['minimum_native_joint_margin_rad'],endpoints=a['endpoints'],
            common_actual_geometry_effector_peak_m=error.tolist() if len(ids)>=99 else None,common_geometry_tail_samples=len(ids),
            audit_sha256=sha(run/'independent_audit.json')))
        env.close()
    assert reference_hashes=={protocol['plan_sha256']} and all(x==source_sets[0] for x in source_sets)
    assert sorted(r['true_height_mm'] for r in rows)==sorted(protocol['true_heights_mm'])
    parent=Path('outputs/highrod_strict_reference_transfer_v1/adapted_bias-3')
    with np.load(parent/'trajectory.npz') as a,np.load(p/'true_height_10mm/trajectory.npz') as b:
        exact={k:bool(np.array_equal(a[k],b[k])) for k in a.files}
    assert all(exact.values()),exact
    rows.sort(key=lambda r:r['true_height_mm'])
    result=dict(results=rows,same_plan_and_sources=True,exact_10mm_replay=exact,protocol_sha256=sha(p/'protocol.json'),
        comparison_sha256=sha(p/'independent_comparison.json'),source_sha256=sha(__file__),common_geometry_reference_sha256=sha(nominal_path),
        scope='Five discrete deterministic heights, not a continuous safe interval or success probability. Common geometry metric: original nominal endpoint hook positions unchanged, feet translated by actual rod offset; supplemental metric only, no relabelling of strict acceptance. No real sensing evaluated.')
    (p/'height_comparison.json').write_text(json.dumps(result,indent=2));(p/'height_comparison_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
