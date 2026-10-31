"""Freeze disjoint local geometry splits and evaluate only the training baseline."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from tools.compare_audited_closure_candidates import load_run, sha
from tools.evaluate_closure_action_bias_v2 import evaluate_parallel
from tools.audit_action_bias_candidate_v2 import audit
from tools.audit_trained_closure_controller_v2 import audit as audit_fixed
from tools.audit_closure_foreaft_reference import audit as audit_reference


def main():
    p=argparse.ArgumentParser()
    for name in ('parent','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];parent=load_run(a.parent)
    assert len(parent['cases'])==16 and all(r['success'] for r in parent['rows'].values())
    m=json.loads((a.parent/'manifest.json').read_text())
    assert m['policy_adapter']=='closure-action-bias-v2'
    known=[]
    for name in ('boundary_combined_old16_20261027_v1','boundary_combined_original8_20261027_v1',
                 'boundary_combined_grid12_20261027_v1','boundary_combined_neighborhood_20261027_v1'):
        known.extend(load_run(Path('outputs')/name)['cases'].values())
    center=parent['cases']['boundary_07'];splits={};seen={json.dumps(g,sort_keys=True) for g in known}
    for split,seed in [('training',2026102702),('validation',2026102703)]:
        rng=np.random.default_rng(seed);u=(np.arange(16)+rng.random(16))/16
        v=(rng.permutation(16)+rng.random(16))/16
        cases={}
        for i,(x,z) in enumerate(zip(u,v)):
            case=copy.deepcopy(center);case['rung_01'][0]+=(2*x-1)*.00075;case['rung_01'][2]+=(2*z-1)*.000075
            key=json.dumps(case,sort_keys=True);assert key not in seen;seen.add(key)
            cases[f'{split}_{i:02d}']=case
        splits[split]=cases
    old_names=['interior_00','interior_04','interior_07','boundary_00','boundary_04','boundary_05','boundary_06','boundary_07']
    training={k:parent['cases'][k] for k in old_names};training.update(splits['training'])
    protocol=dict(parent=str(a.parent.resolve()),parent_audit_sha256=parent['audit_sha256'],parent_checkpoint_sha256=parent['checkpoint_sha256'],
        training_cases=training,validation_cases=splits['validation'],center=center,half_width_x_z_m=[.00075,.000075],
        training_geometry_seed=2026102702,validation_geometry_seed=2026102703,
        planned_training_seed=2026102704,planned_additional_steps=131072,reward_mode='average',quality_bonus_weight=.1,
        calibration_fixed=True,source_sha256=sha(__file__),
        scope='Prospectively frozen local geometry split for a future average-reward curriculum from the already-audited 32768-step average arm. Eight known guard cases plus sixteen new training geometries; sixteen distinct new validation geometries withheld until the final checkpoint is frozen. Geometry region chosen after observing prior local failures. No broad-random-wall or independent-scene generalization claim; no acceptance or calibration changes; no early checkpoint selection on validation outcomes.')
    a.protocol.mkdir(exist_ok=False);(a.protocol/'protocol.json').write_text(json.dumps(protocol,indent=2));(a.protocol/'source.py').write_bytes(Path(__file__).read_bytes())
    a.output.mkdir(exist_ok=False);shutil.copytree(a.parent/'source_snapshot',a.output/'source_snapshot')
    shutil.copy2(a.parent/'policy.zip',a.output/'policy.zip');shutil.copy2(m['prefix_checkpoint'],a.output/'prefix_policy.zip')
    sources=dict(m['sources'])
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if not f:continue
        f=Path(f).resolve()
        if f.suffix!='.py' or not f.is_relative_to(root):continue
        rel=f.relative_to(root)
        if rel.parts[0] not in ('training','tools'):continue
        dest=a.output/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes());sources[str(rel)]=sha(f)
    m.update(sources=sources,parent_experiment=str(a.parent.resolve()),parent_manifest_sha256=sha(a.parent/'manifest.json'),
        prefix_checkpoint=str((a.output/'prefix_policy.zip').resolve()),evaluation_cases=training,training_cases={},evaluation_only=True,
        new_training_steps=0,require_inactive_case_exact_parent=False,geometry_protocol_sha256=sha(a.protocol/'protocol.json'),
        scope=protocol['scope']+' Baseline only, before new curriculum training. Eight existing complete guard trajectories reused by hash; sixteen new training cases run from physical time zero. Validation cases are not evaluated.')
    (a.output/'manifest.json').write_text(json.dumps(m,indent=2))
    summary=dict(evaluation_pending=True,checkpoint_sha256=sha(a.output/'policy.zip'),evaluation_only=True,new_training_steps=0)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    previous=json.loads((a.parent/'summary.json').read_text())['after'];rows=[];reused=[]
    for case in old_names:
        row=next(r for r in previous if r['case']==case);f=a.parent/f'after_{case}.npz';assert sha(f)==row['trajectory_sha256']
        shutil.copy2(f,a.output/f.name);rows.append(row);reused.append(dict(case=case,sha256=sha(f)))
    new=evaluate_parallel(a.output,a.output/'parallel_after',list(splits['training']),workers=4)['trajectories']
    for row in new:
        f=f'after_{row["case"]}.npz';shutil.copy2(a.output/'parallel_after'/f,a.output/f)
    rows+=new;summary.update(after=rows,evaluation_pending=False,reused_guard_trajectories=reused)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));(a.output/'after.json').write_text(json.dumps(rows,indent=2))
    audit(a.output);audit_fixed(a.output);audit_reference(a.output)
    verified=load_run(a.output)
    result=dict(completed=True,training_baseline_successes=sum(r['success'] for r in verified['rows'].values()),
        training_cases=24,new_training_geometries=16,reused_guard_cases=8,validation_evaluated=False,
        protocol_sha256=sha(a.protocol/'protocol.json'),audit_sha256=verified['audit_sha256'])
    (a.output/'baseline_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
