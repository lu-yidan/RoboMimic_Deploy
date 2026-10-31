"""Prove the initialized feedback actor preserves the existing full-cycle baseline."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from stable_baselines3 import PPO
from tools.compare_audited_closure_candidates import load_run, sha
from tools.evaluate_seeded_reference_feedback import evaluate_parallel, FEEDBACK_CONTRACT
from tools.audit_seeded_reference_feedback import audit
from tools.audit_closure_candidate_prefix import audit as audit_prefix
from tools.audit_trained_closure_controller import audit as audit_fixed


def main():
    p = argparse.ArgumentParser()
    for name in ('parent', 'seed', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--cases', nargs='+', default=['foot_z_plus15'])
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    verified = load_run(a.parent)
    m = json.loads((a.parent/'manifest.json').read_text())
    seed = json.loads((a.seed/'summary.json').read_text())
    protocol = json.loads((a.seed/'protocol.json').read_text())
    assert seed['passed'] and seed['zero_mean_exact']
    assert seed['checkpoint_sha256'] == sha(a.seed/'policy_initial.zip')
    assert seed['protocol_sha256'] == sha(a.seed/'protocol.json')
    assert protocol['parent_audit_sha256'] == verified['audit_sha256']
    assert len(set(a.cases)) == len(a.cases) and set(a.cases) <= set(m['evaluation_cases'])
    teacher = PPO.load(a.parent/'policy.zip', device='cpu')
    feedback = PPO.load(a.seed/'policy_initial.zip', device='cpu')
    assert teacher.observation_space.shape == (148,) and feedback.observation_space.shape == (153,)
    sources = dict(m['sources'])
    sources.update(protocol['sources'])
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if not name:
            continue
        file = Path(name).resolve()
        if file.suffix == '.py' and file.is_relative_to(root):
            relative = file.relative_to(root)
            if relative.parts[0] in ('tools', 'training'):
                sources[str(relative)] = sha(file)
    a.output.mkdir(exist_ok=False)
    for name, digest in sources.items():
        assert sha(root/name) == digest
        dest = a.output/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((root/name).read_bytes())
    shutil.copy2(a.seed/'policy_initial.zip', a.output/'policy.zip')
    shutil.copy2(a.parent/'policy.zip', a.output/'teacher.zip')
    shutil.copy2(m['prefix_checkpoint'], a.output/'prefix_policy.zip')
    m.update(sources=sources, teacher_config=m['config'], config=protocol['config'],
        teacher_observation_contract=m['observation_contract'], observation_contract=FEEDBACK_CONTRACT,
        teacher_checkpoint=str((a.output/'teacher.zip').resolve()), teacher_checkpoint_sha256=verified['checkpoint_sha256'],
        prefix_checkpoint=str((a.output/'prefix_policy.zip').resolve()),
        evaluation_cases={k: m['evaluation_cases'][k] for k in a.cases}, training_cases={},
        parent_experiment=str(a.parent.resolve()), parent_manifest_sha256=sha(a.parent/'manifest.json'),
        feedback_initialization=str(a.seed.resolve()), feedback_seed_protocol_sha256=sha(a.seed/'protocol.json'),
        feedback_delta_scale=protocol['delta_scale'], feedback_hold_steps=protocol['hold_steps'],
        feedback_physical_gamma=protocol['physical_gamma'], reference_command_mode='learned_delta',
        evaluation_only=True, new_training_steps=0, require_inactive_case_exact_parent=True,
        scope='Zero initialized 153-input, five-delta, 10 Hz reference actor over a frozen 148-input teacher. Full time-zero equivalence test, no new training or performance improvement.')
    (a.output/'manifest.json').write_text(json.dumps(m, indent=2))
    s = dict(checkpoint_sha256=sha(a.output/'policy.zip'), evaluation_pending=True,
             evaluation_only=True, new_training_steps=0)
    (a.output/'summary.json').write_text(json.dumps(s, indent=2))
    rows = evaluate_parallel(a.output, a.output/'parallel_after', workers=1)['trajectories']
    for row in rows:
        name = f"after_{row['case']}.npz"
        shutil.copy2(a.output/'parallel_after'/name, a.output/name)
    s.update(after=rows, evaluation_pending=False)
    (a.output/'summary.json').write_text(json.dumps(s, indent=2))
    (a.output/'after.json').write_text(json.dumps(rows, indent=2))
    audit(a.output)
    audit_prefix(a.output)
    audit_fixed(a.output)
    checks = []
    for row in rows:
        name = f"after_{row['case']}.npz"
        with np.load(a.output/name) as current, np.load(a.parent/name) as parent:
            exact = {key: bool(np.array_equal(current[key], parent[key])) for key in
                ('qpos', 'qvel', 'time_s', 'observations', 'actions', 'torques', 'rewards',
                 'waist_reference_trace', 'body_reference_trace', 'adaptive_reference_trace')}
            exact['zero_feedback_actions'] = bool(np.array_equal(current['feedback_actions'], np.zeros_like(current['feedback_actions'])))
        exact['same_task_success'] = bool(row['success']) == verified['rows'][row['case']]['success']
        checks.append(dict(case=row['case'], passed=all(exact.values()), exact=exact))
    result = dict(passed=all(r['passed'] for r in checks), rows=checks,
        seed_summary_sha256=sha(a.seed/'summary.json'), parent_audit_sha256=verified['audit_sha256'],
        manifest_sha256=sha(a.output/'manifest.json'), summary_sha256=sha(a.output/'summary.json'),
        source_sha256=sha(__file__), scope='Untrained zero-delta reference actor reproduces the parent complete trajectory exactly. No learned-policy success claim.')
    (a.output/'zero_feedback_gate.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    assert result['passed']


if __name__ == '__main__':
    main()
