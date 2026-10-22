"""One fixed motor-only candidate from static IK, tested on all five heights."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import sys
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha

ROOT=Path(__file__).resolve().parents[1]


def envelope(t):
    enter=float(np.clip((t-57.)/2.,0.,1.));leave=float(np.clip((65.-t)/2.,0.,1.))
    return min(enter*enter*(3-2*enter),leave*leave*(3-2*leave))


def run(job):
    output,case,offsets,vector,*base_modifier=job;p=Path(output)/case;p.mkdir(exist_ok=False)
    plan=Path('outputs/highrod_endpoint_adapt_plan_v1/plan.npz')
    if base_modifier:
        from training.temporal_base_reference_env import TemporalBaseReferenceEnv
        env=TemporalBaseReferenceEnv(plan=plan,cases={case:offsets},normal_bias_y_deg=-3.,base_bias=base_modifier[0])
    else:env=FullCycleResidualEnv(plan=plan,cases={case:offsets},normal_bias_y_deg=-3.)
    obs,_=env.reset(options={'case':case});e=env.env;sources={}
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if not path:continue
        path=Path(path).resolve()
        if path.suffix!='.py' or not path.is_relative_to(ROOT):continue
        rel=path.relative_to(ROOT)
        if rel.parts[0] not in ('training','tools'):continue
        dest=p/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());sources[str(rel)]=sha(dest)
    manifest=dict(label=case,case=case,offsets=offsets,reference=str(plan.resolve()),reference_sha256=sha(plan),bundle=str(e.bundle),
        bundle_manifest_sha256=sha(e.bundle/'manifest.json'),normal_bias_y_deg=-3.,memory_enabled=True,contract=env.contract,horizon_s=env.horizon,
        sources=sources,protocol_sha256=sha(Path(output)/'protocol.json'),action_rule=dict(type='fixed_vector_cubic_envelope',vector=vector,start_s=57.,end_s=65.,ramp_s=2.),
        scope='Fixed26motor residual candidate. Only joint components of offline IK are used; root translation is NOT injected. Actual-pose IK does not guarantee reference-action efficacy. All five known height cases retained.')
    if base_modifier:
        manifest['reference_modifier']=dict(type='bounded-base-reference-57to65s-cubic2s-v1',base_bias=base_modifier[0],start_s=57.,end_s=65.,ramp_s=2.)
        manifest['scope']='Same26motor joint residual plus temporary bounded base TRACKING REFERENCE. No direct state overwrite or applied external force. Endpoint targets remain unchanged; separate reference ablation.'
    (p/'manifest.json').write_text(json.dumps(manifest,indent=2))
    q=[e.data.qpos.copy()];v=[e.data.qvel.copy()];t=[float(e.data.time)];observations=[];actions=[];torques=[];rewards=[]
    try:
        while True:
            action=envelope(float(e.data.time))*np.asarray(vector);observations.append(obs.copy());actions.append(action.copy())
            obs,reward,done,truncated,info=env.step(action)
            q.append(e.data.qpos.copy());v.append(e.data.qvel.copy());t.append(float(e.data.time));torques.append(env.last_torque.copy());rewards.append(reward)
            if done or truncated:break
        np.savez_compressed(p/'trajectory.npz',qpos=q,qvel=v,time_s=t,observations=observations,actions=actions,torques=torques,rewards=rewards)
        result=dict(info,label=case,trajectory_sha256=sha(p/'trajectory.npz'),manifest_sha256=sha(p/'manifest.json'))
        (p/'summary.json').write_text(json.dumps(result,indent=2));return {k:result[k] for k in ('label','success','reason','time_s')}
    finally:env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--base-reference',action='store_true');args=parser.parse_args()
    source=Path('outputs/highrod_limit_posture_locked_waist_v1');audit=json.loads((source/'independent_audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256']==sha(source/'summary.json')
    env=FullCycleResidualEnv();offsets=[];base_offsets=[];candidate_hashes={}
    for label in ('true_height_12.5mm','true_height_15mm'):
        file=source/f'{label}_seed1.npz'
        record=next(r for r in audit['results'] if r['case']==label.replace('true_height_','foot_z_plus').replace('mm','') and r['seed']==1)
        assert record['candidate_sha256']==sha(file)
        with np.load(file) as z:
            difference=z['candidate_qpos']-z['original_qpos'];offsets.append(difference[env.env.qadr[env.indices]]);base_offsets.append(difference[:3])
        candidate_hashes[str(file)]=sha(file)
    env.close();vector=np.mean(offsets,axis=0)/.015
    assert np.isfinite(vector).all() and abs(vector).max()<=1.
    cases={f'foot_z_plus{h:g}':{'rung_01':[0.,0.,h/1000]} for h in (5.,7.5,10.,12.5,15.)}
    args.output.mkdir(exist_ok=False)
    jobs=[(str(args.output),case,offset,vector.tolist()) for case,offset in cases.items()]
    if args.base_reference:
        baseline=Path('outputs/load_margin_fixed_residual_v1');previous=json.loads((baseline/'protocol.json').read_text());selection=json.loads((baseline/'selection.json').read_text())
        assert not selection['selected'] and selection['protocol_sha256']==sha(baseline/'protocol.json') and np.array_equal(vector,previous['normalized_action_vector'])
        base_bias=np.mean(base_offsets,axis=0).tolist();jobs=[(*job,base_bias) for job in jobs]
    protocol=dict(jobs=jobs,cases=cases,normalized_action_vector=vector.tolist(),candidate_hashes=candidate_hashes,
        source_audit_sha256=sha(source/'independent_audit.json'),source_sha256=sha(__file__),
        selection='One common vector: average of locked-waist seed1 joint deltas (smaller maximum joint deviation in both known failures). No per-case action tuning. Reject if no strict failures repaired or any successful baseline regresses.',
        scope='Calibration using known failed poses, not held-out evaluation. Smooth57..65s motor-reference action only, no base command or state overwrite. Preserve all outcomes; compare to zero baseline3/5.')
    if args.base_reference:
        protocol.update(base_reference_bias_m=base_bias,joint_baseline_selection_sha256=sha(baseline/'selection.json'),
            scope='Known-case ablation: add mean static IK base shift as57..65s tracking reference, with identical joint residual to rejected joint-only baseline. Existing torque controller realizes motion; no direct base actuation/state modification. No generalization claim.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2));(args.output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            r=future.result();results.append(r);(args.output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)


if __name__=='__main__':main()
