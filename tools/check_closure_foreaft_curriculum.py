"""Check five-action zero feedback against audited complete teacher trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.closure_foreaft_curriculum_env import ClosureForeaftCurriculumEnv
from tools.compare_audited_closure_candidates import load_run,sha


def main():
    parser=argparse.ArgumentParser()
    for name in ('parent','cache','zero-gate','probe-gate','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    verified=load_run(args.parent);manifest=json.loads((args.parent/'manifest.json').read_text())
    for path in (args.zero_gate,args.probe_gate):
        gate=json.loads((path/'summary.json').read_text());assert gate['passed']
        assert gate['sources']['training/closure_foreaft_reference.py']==sha('training/closure_foreaft_reference.py')
        assert gate['trajectory_sha256']==sha(path/'trajectory.npz')
    cache=json.loads((args.cache/'protocol.json').read_text());assert cache['cases']==manifest['evaluation_cases']
    args.output.mkdir(exist_ok=False)
    env=ClosureForeaftCurriculumEnv(args.cache,controller_modifier=manifest['controller_modifier'],
        teacher_checkpoint=args.parent/'policy.zip',teacher_sha256=sha(args.parent/'policy.zip'),record_dir=args.output/'records')
    rows=[]
    try:
        assert env.action_space.shape==(5,) and env.observation_space.shape==(153,)
        for case in manifest['evaluation_cases']:
            obs,_=env.reset(options={'case':case});base=env.env.unwrapped
            q=[base.env.data.qpos.copy()];v=[base.env.data.qvel.copy()];observations=[]
            while True:
                assert obs.shape==(153,) and np.isfinite(obs).all() and np.array_equal(obs[-5:],np.zeros(5))
                observations.append(obs[:148].copy())
                obs,_,done,truncated,info=env.step(np.zeros(5,dtype=np.float32))
                q.append(base.env.data.qpos.copy());v.append(base.env.data.qvel.copy())
                if done or truncated:break
            with np.load(args.parent/f'after_{case}.npz') as original:
                differences=dict(qpos=float(abs(original['qpos'][4983:]-q).max()),qvel=float(abs(original['qvel'][4983:]-v).max()),
                    observations=float(abs(original['observations'][4983:]-observations).max()))
            passed=differences['qpos']<1e-10 and differences['qvel']<1e-8 and differences['observations']<1e-6 and info['success']==verified['rows'][case]['success']
            row=dict(case=case,passed=bool(passed),differences=differences,teacher_task_success=info['success']);rows.append(row);print(json.dumps(row),flush=True)
            assert passed
        records=list((args.output/'records').glob('*.npz'));assert len(records)==1
        with np.load(records[0]) as trace:
            recording_checks=dict(teacher_actions=trace['actions'].shape==(700,26),teacher_observations=trace['observations'].shape==(700,148),
                feedback_actions=trace['adaptive_actions'].shape==(700,5),feedback_observations=trace['adaptive_observations'].shape==(700,153),
                feedback_commands=trace['adaptive_reference_trace'].shape==(2800,21))
        sources={p:sha(p) for p in ('training/closure_foreaft_reference.py','training/closure_foreaft_curriculum_env.py','tools/check_closure_foreaft_curriculum.py')}
        result=dict(passed=all(r['passed'] for r in rows) and all(recording_checks.values()),rows=rows,recording_checks=recording_checks,
            parent_audit_sha256=verified['audit_sha256'],teacher_sha256=sha(args.parent/'policy.zip'),cache_summary_sha256=sha(args.cache/'summary.json'),
            zero_gate_sha256=sha(args.zero_gate/'summary.json'),probe_gate_sha256=sha(args.probe_gate/'summary.json'),sources=sources,
            scope='Zero feedback reproduces the fixed teacher including its failures. This verifies routing and reset semantics, not five-action learned performance.')
        (args.output/'summary.json').write_text(json.dumps(result,indent=2))
        for name in sources:
            target=args.output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes())
        assert result['passed']
    finally:env.close()


if __name__=='__main__':main()
