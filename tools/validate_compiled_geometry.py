"""Explicit single-cycle/prefix audit entry point for fully compiled rod scenes.

Existing evaluators and auditors are reused with declared environment adapters
inside each fresh process. Their source files and the adopted baseline remain
unchanged. No process-wide subprocess interception or hidden environment flag.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
from training.compiled_rung_geometry import CompiledStrictCycleEnv,reset_compiled_geometry
from tools import validate_swing_reference as single
from tools import validate_frozen_v2_geometry as frozen
from tools.validate_swing_reference import ROOT,read,write,sha

MODE='complete_mjspec_model_at_reset_v1'
SCENES=[]


def capture(base):
    SCENES.append(dict(case=base.case,variant=base.variant,time_s=float(base.env.data.time)))


class ObservedCompiledStrictEnv(single.ObservedStrictEnv):
    def reset(self,**kwargs):
        self.physical_tail=[];self.strict_tail=deque()
        result=reset_compiled_geometry(self,**kwargs);capture(self);return result

    def info(self,reason,success):
        return dict(super().info(reason,success),scene_compiler=MODE,
                    scene_geometry_sha256=self.variant['geometry_sha256'])


class PrefixCompiledStrictEnv(CompiledStrictCycleEnv):
    def reset(self,**kwargs):
        result=super().reset(**kwargs);capture(self);return result


def install_adapters():
    import training.live_prefix_landing_env as prefix
    import tools.audit_action_bias_candidate_v2 as audit_policy
    single.ObservedStrictEnv=ObservedCompiledStrictEnv
    prefix.FullCycleResidualEnv=PrefixCompiledStrictEnv
    audit_policy.FullCycleResidualEnv=CompiledStrictCycleEnv


def check_manifest(out):
    manifest=read(out/'manifest.json')
    assert manifest['scene_compiler']==MODE
    for relative in ('tools/validate_compiled_geometry.py','training/compiled_rung_geometry.py'):
        assert sha(ROOT/relative)==manifest['sources'][relative]==sha(out/'source_snapshot'/relative)
    assert read(out/'compiled_preparation_audit.json')['manifest_sha256']==sha(out/'manifest.json')
    return manifest


def prepare(hook_protocol,geometry_protocol,out):
    hook_protocol=hook_protocol.resolve();declared=read(hook_protocol)
    candidate=Path(declared['candidate']);geometry=read(geometry_protocol)
    assert declared['status']=='protocol_only_no_new_grid_rollouts'
    assert geometry['validation_cases']==declared['cases']
    assert geometry['original_protocol_sha256']==sha(hook_protocol)
    assert sha(candidate/'manifest.json')==declared['candidate_manifest_sha256']
    frozen.prepare(candidate,geometry_protocol,out)
    protocol=read(out/'validation_protocol.json');manifest=read(out/'manifest.json')
    role='Frozen local rung_06 X/Z development grid around T12; not an unseen or random-wall generalization test.'
    shutil.copy2(hook_protocol,out/'hook_uncertainty_protocol.json')
    protocol.update(case_role=role,scene_compiler=MODE,hook_protocol_sha256=sha(hook_protocol),
                    compiled_pipeline_sha256=sha(__file__))
    write(out/'validation_protocol.json',protocol)
    for relative in ('training/compiled_rung_geometry.py','tools/validate_compiled_geometry.py'):
        target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target);manifest['sources'][relative]=sha(target)
    manifest.update(scene_compiler=MODE,case_role=role,validation_protocol_sha256=sha(out/'validation_protocol.json'),
        hook_protocol_sha256=sha(hook_protocol),
        execution_adapters={'single_evaluator':'ObservedCompiledStrictEnv',
                            'independent_prefix':'PrefixCompiledStrictEnv','state_policy_auditor':'CompiledStrictCycleEnv'},
        scope=role+' Complete MuJoCo model compilation at reset replaces the old partial BVH update. Robot dynamics and frozen controllers unchanged; tiny compiler geometry roundoff is retained and reported. Single-cycle screen only, no training or promotion.')
    write(out/'manifest.json',manifest)
    write(out/'compiled_preparation_audit.json',dict(passed=True,rollouts_started=False,cases=len(declared['cases']),
        manifest_sha256=sha(out/'manifest.json'),protocol_sha256=sha(out/'validation_protocol.json'),
        hook_protocol_sha256=sha(hook_protocol),source_sha256=sha(__file__),scene_compiler=MODE))
    print(json.dumps(dict(prepared=True,output=str(out),cases=len(declared['cases']),scene_compiler=MODE)),flush=True)


def worker(out,case,kind):
    manifest=check_manifest(out);assert case in manifest['evaluation_cases']
    install_adapters();single.worker(out,case,kind)
    assert len(SCENES)==1 and SCENES[0]['case']==case and SCENES[0]['time_s']==0.
    if kind=='eval':
        target=out/'eval'/case/'compiled_scene_execution.json'
        trajectory=out/'eval'/case/f'after_{case}.npz'
    else:
        target=out/'prefix'/f'{case}_compiled_scene.json';trajectory=out/'prefix'/f'{case}.npz'
    write(target,dict(passed=True,case=case,kind=kind,scene_compiler=MODE,physical_resets=1,
        scene=SCENES[0]['variant'],trajectory_sha256=sha(trajectory),manifest_sha256=sha(out/'manifest.json'),source_sha256=sha(__file__)))


def audit(out):
    manifest=check_manifest(out);install_adapters()
    single.audit(out);chain=frozen.audit(out)
    rows=[]
    env=CompiledStrictCycleEnv(plan=manifest['reference'],cases=manifest['evaluation_cases'],
        normal_bias_y_deg=manifest['normal_bias_y_deg'],memory_enabled=manifest['force_history_memory'])
    try:
        for case in manifest['evaluation_cases']:
            env.reset(options={'case':case});expected=env.variant['geometry_sha256']
            result=read(out/'eval'/case/'result.json')
            kinds=['eval']+(['prefix'] if result['time_s']>=70.2-1e-7 else [])
            checks={}
            for kind in kinds:
                path=out/'eval'/case/'compiled_scene_execution.json' if kind=='eval' else out/'prefix'/f'{case}_compiled_scene.json'
                evidence=read(path)
                trajectory=out/f'after_{case}.npz' if kind=='eval' else out/'prefix'/f'{case}.npz'
                checks[kind]=bool(evidence['passed'] and evidence['physical_resets']==1
                    and evidence['case']==case and evidence['kind']==kind and evidence['scene_compiler']==MODE
                    and evidence['manifest_sha256']==sha(out/'manifest.json')
                    and evidence['trajectory_sha256']==sha(trajectory)
                    and evidence['source_sha256']==sha(__file__)
                    and evidence['scene']['geometry_sha256']==expected)
            rows.append(dict(case=case,passed=all(checks.values()),checks=checks,geometry_sha256=expected))
    finally:env.close()
    report=dict(passed=chain['passed'] and all(r['passed'] for r in rows),
        eligible_single_cycle_reference=chain['eligible_single_cycle_reference'] and all(r['passed'] for r in rows),
        physical_v2_count=chain['physical_v2_count'],cases=len(rows),rows=rows,scene_compiler=MODE,
        base_chain_audit_sha256=sha(out/'reference_chain_audit.json'),manifest_sha256=sha(out/'manifest.json'),
        source_sha256=sha(__file__),new_training_steps=0,
        scope='Compiled-scene execution binding plus the existing independent single-cycle physical, controller, actor and prefix audits. No two-cycle result or controller comparison yet.')
    write(out/'compiled_reference_chain_audit.json',report)
    (out/'compiled_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


def run(out,workers):
    manifest=check_manifest(out);cases=list(manifest['evaluation_cases'])
    def stage(kind,names):
        (out/kind).mkdir(exist_ok=True)
        def one(case):
            with (out/kind/f'{case}.log').open('w',encoding='utf8') as stream:
                subprocess.run([sys.executable,'-m','tools.validate_compiled_geometry','worker',
                    '--output',str(out),'--case',case,'--kind',kind],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
            print(json.dumps(dict(case=case,kind=kind,worker_finished=True)),flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(one,c) for c in names]):future.result()
    assert not (out/'eval').exists() and not (out/'prefix').exists()
    stage('eval',cases)
    rows=[]
    for case in cases:
        result=read(out/'eval'/case/'result.json');trajectory=out/'eval'/case/f'after_{case}.npz'
        assert sha(trajectory)==result['trajectory_sha256']
        shutil.copy2(trajectory,out/trajectory.name);rows.append(result)
    write(out/'after.json',rows)
    write(out/'summary.json',dict(evaluation_pending=False,after=rows,checkpoint_sha256=sha(out/'policy.zip'),additional_steps=0))
    stage('prefix',[r['case'] for r in rows if r['time_s']>=70.2-1e-7])
    audit(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run','worker','audit'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--hook-protocol',type=Path)
    parser.add_argument('--geometry-protocol',type=Path);parser.add_argument('--workers',type=int,choices=(1,2,4),default=4)
    parser.add_argument('--case');parser.add_argument('--kind',choices=('eval','prefix'))
    args=parser.parse_args();out=args.output.resolve()
    if args.command=='prepare':prepare(args.hook_protocol,args.geometry_protocol,out)
    elif args.command=='run':run(out,args.workers)
    elif args.command=='worker':worker(out,args.case,args.kind)
    else:audit(out)
