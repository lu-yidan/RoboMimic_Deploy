"""Matched five-delta PPO with a separately calibrated, freshly validated critic."""
import argparse
from functools import partial
import json
from pathlib import Path
import shutil
import sys
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from training.seeded_reference_delta_env import SeededReferenceDeltaEnv
from training.held_reference_command import HeldReferenceCommand
from training.balanced_case_reset import BalancedCaseReset
from tools.compare_audited_closure_candidates import load_run, sha
from tools.continue_vector_full_cycle_training import VectorProgress, state_difference
from tools.evaluate_seeded_reference_feedback import evaluate_parallel, FEEDBACK_CONTRACT
from tools.audit_seeded_reference_feedback import audit
from tools.audit_closure_candidate_prefix import audit as audit_prefix
from tools.audit_trained_closure_controller import audit as audit_fixed


def make_env(protocol, output, rank):
    torch.set_num_threads(1)
    env = SeededReferenceDeltaEnv(Path(protocol['worker_shards'])/f'worker_{rank}',
        controller_modifier=protocol['controller_modifier'],
        teacher_checkpoint=protocol['teacher_checkpoint'], teacher_sha256=protocol['teacher_sha256'],
        seed_command=protocol['seed_command'], delta_scale=protocol['delta_scale'], weight=.1)
    held = HeldReferenceCommand(env, repeat=protocol['hold_steps'], physical_gamma=protocol['physical_gamma'])
    return Monitor(BalancedCaseReset(held), str(Path(output)/f'episodes_worker_{rank}'))


class FeedbackProgress(VectorProgress):
    def __init__(self, output, parent_steps):
        super().__init__(output, parent_steps)
        self.teacher_steps = 0

    def _on_step(self):
        self.teacher_steps += sum(int(i['macro_executed_teacher_steps']) for i in self.locals['infos'])
        result = super()._on_step()
        if (self.num_timesteps-self.parent_steps) % 512 == 0:
            (self.output/'feedback_progress.json').write_text(json.dumps(dict(
                macro_steps=self.num_timesteps-self.parent_steps, teacher_steps=self.teacher_steps,
                completed_episodes=len(self.episodes),
                successful_episodes=sum(bool(r['success']) for r in self.episodes)), indent=2))
        return result


def main():
    p = argparse.ArgumentParser()
    for name in ('seed', 'zero-gate', 'negative-gate', 'environment-gate', 'critic', 'fresh-critic-gate', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=19456)
    a = p.parse_args()
    assert a.steps > 0 and a.steps % 512 == 0
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((a.seed/'protocol.json').read_text())
    seed = json.loads((a.seed/'summary.json').read_text())
    assert seed['passed'] and seed['checkpoint_sha256'] == sha(a.seed/'policy_initial.zip')
    assert seed['protocol_sha256'] == sha(a.seed/'protocol.json')
    calibration = json.loads((a.critic/'summary.json').read_text())
    calibration_protocol = json.loads((a.critic/'protocol.json').read_text())
    fresh = json.loads((a.fresh_critic_gate/'summary.json').read_text())
    fresh_protocol = json.loads((a.fresh_critic_gate/'protocol.json').read_text())
    assert calibration['passed'] and calibration['frozen_actor_and_distribution_exact']
    assert fresh['passed'] and fresh['frozen_actor_and_distribution_exact']
    assert calibration['checkpoint_sha256'] == sha(a.critic/'policy_initial.zip')
    assert calibration['protocol_sha256'] == sha(a.critic/'protocol.json')
    assert calibration_protocol['seed_checkpoint_sha256'] == seed['checkpoint_sha256']
    assert fresh['protocol_sha256'] == sha(a.fresh_critic_gate/'protocol.json')
    assert fresh['critic_checkpoint_sha256'] == calibration['checkpoint_sha256']
    assert fresh['critic_summary_sha256'] == sha(a.critic/'summary.json')
    assert fresh_protocol['initial_checkpoint_sha256'] == seed['checkpoint_sha256']
    for source, digest in calibration_protocol['sources'].items():
        assert sha(source) == digest
    for path, digest in calibration_protocol['evidence'].items():
        assert sha(path) == digest
    for source, digest in fresh_protocol['sources'].items():
        assert sha(source) == digest
    for row in fresh['rows']:
        assert sha(a.fresh_critic_gate/row['file']) == row['trajectory_sha256']
    warm = PPO.load(a.critic/'policy_initial.zip', device='cpu')
    raw = PPO.load(a.seed/'policy_initial.zip', device='cpu')
    frozen = [key for key in raw.policy.state_dict() if not key.startswith(('mlp_extractor.value_net.', 'value_net.'))]
    assert all(torch.equal(raw.policy.state_dict()[key], warm.policy.state_dict()[key]) for key in frozen)
    parent = Path(protocol['parent'])
    verified = load_run(parent)
    assert verified['audit_sha256'] == protocol['parent_audit_sha256']
    assert sha(protocol['teacher_checkpoint']) == protocol['teacher_sha256']
    assert sha(Path(protocol['worker_shards'])/'summary.json') == protocol['shards_summary_sha256']
    gates = {str(a.zero_gate/'zero_feedback_gate.json'): None,
             str(a.negative_gate/'summary.json'): None,
             str(a.environment_gate/'summary.json'): None,
             str(a.critic/'summary.json'): None,
             str(a.fresh_critic_gate/'summary.json'): None}
    for path in gates:
        assert json.loads(Path(path).read_text())['passed']
        gates[path] = sha(path)
    zero = json.loads((a.zero_gate/'zero_feedback_gate.json').read_text())
    negative = json.loads((a.negative_gate/'summary.json').read_text())
    environment = json.loads((a.environment_gate/'summary.json').read_text())
    assert zero['seed_summary_sha256'] == sha(a.seed/'summary.json')
    assert zero['parent_audit_sha256'] == verified['audit_sha256']
    assert negative['zero_gate_sha256'] == sha(a.zero_gate/'zero_feedback_gate.json')
    assert negative['auditor_sha256'] == sha(root/'tools/audit_feedback_reference_commands.py')
    assert environment['parent_audit_sha256'] == verified['audit_sha256']
    assert sha(a.environment_gate/'summary.json') == protocol['environment_gate_sha256']
    for source, digest in protocol['sources'].items():
        assert sha(root/source) == digest
    m = json.loads((a.zero_gate/'manifest.json').read_text())
    sources = dict(m['sources'])
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
    for source, digest in sources.items():
        assert sha(root/source) == digest
        dest = a.output/'source_snapshot'/source
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((root/source).read_bytes())
    shutil.copy2(protocol['teacher_checkpoint'], a.output/'teacher.zip')
    shutil.copy2(m['prefix_checkpoint'], a.output/'prefix_policy.zip')
    shutil.copy2(a.seed/'protocol.json', a.output/'initialization_protocol.json')
    m.update(sources=sources, evaluation_cases=protocol['cases'], training_cases=protocol['cases'],
        teacher_checkpoint=str((a.output/'teacher.zip').resolve()),
        prefix_checkpoint=str((a.output/'prefix_policy.zip').resolve()),
        evaluation_only=False, require_inactive_case_exact_parent=False, environments=4,
        feedback_initialization=str(a.seed.resolve()), initial_checkpoint_sha256=calibration['checkpoint_sha256'],
        observation_contract=FEEDBACK_CONTRACT, feedback_training_macro_steps=a.steps,
        maximum_teacher_transitions=a.steps*protocol['hold_steps'], training_episode_seconds=14.,
        initial_critic='value-only calibration, followed by eight fresh stochastic validation episodes',
        critic_initialization=str(a.critic.resolve()), critic_summary_sha256=sha(a.critic/'summary.json'),
        fresh_critic_gate=str(a.fresh_critic_gate.resolve()), gates=gates,
        scope='Exploratory PPO of five bounded reference deltas at 10 Hz in the final 14 seconds. '
              'Same zero-mean feedback actor, distribution, PPO settings, seed and budget as the fresh-critic run; only value initialization differs. '
              'Frozen 26D teacher; all task/physics limits unchanged. Different action authority, '
              'decision frequency, training window and initialization from 26D PPO; not a matched ablation. '
              'Training starts from cached states; evaluation starts at time zero. No automatic promotion.')
    m.pop('new_training_steps', None)
    # Prioritize the known failure and thinnest margin without changing training.
    priority = ['z15.1', 'x+0.1_z15', 'foot_z_plus15', 'x-0.1_z15',
                'foot_z_plus12.5', 'foot_z_plus10', 'foot_z_plus5', 'foot_z_plus7.5']
    assert set(priority) == set(m['evaluation_cases'])
    m['evaluation_cases'] = {case: m['evaluation_cases'][case] for case in priority}
    (a.output/'manifest.json').write_text(json.dumps(m, indent=2))
    env = SubprocVecEnv([partial(make_env, protocol, str(a.output.resolve()), rank)
                        for rank in range(4)], start_method='spawn')
    model = PPO.load(a.critic/'policy_initial.zip', env=env, device='cpu', force_reset=True)
    initial = PPO.load(a.critic/'policy_initial.zip', device='cpu')
    assert state_difference(model.policy.state_dict(), initial.policy.state_dict()) == 0
    assert state_difference(model.policy.optimizer.state_dict(), initial.policy.optimizer.state_dict()) == 0
    for key in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'target_kl'):
        assert getattr(model, key) == protocol['config'][key]
    model.set_random_seed(protocol['config']['seed'])
    model.set_logger(configure(str(a.output), ['csv']))
    start_steps = int(model.num_timesteps)
    callback = FeedbackProgress(a.output, start_steps)
    model.save(a.output/'policy_before')
    started = time.perf_counter()
    try:
        model.learn(total_timesteps=a.steps, reset_num_timesteps=False, callback=callback)
    except Exception as exc:
        model.save(a.output/'policy_interrupted')
        (a.output/'interrupted_status.json').write_text(json.dumps(dict(reason=str(exc),
            macro_steps=int(model.num_timesteps-start_steps)), indent=2))
        raise
    finally:
        env.close()
    model.save(a.output/'policy.zip')
    result = dict(checkpoint_sha256=sha(a.output/'policy.zip'),
        additional_macro_steps=int(model.num_timesteps-start_steps), teacher_steps=callback.teacher_steps,
        training_wall_seconds=time.perf_counter()-started, completed_training_episodes=len(callback.episodes),
        successful_training_episodes=sum(bool(r['success']) for r in callback.episodes),
        case_samples=callback.case_samples, evaluation_pending=True,
        teacher_unchanged=sha(a.output/'teacher.zip') == protocol['teacher_sha256'])
    assert result['teacher_unchanged']
    if (a.output/'STOP_REQUESTED').exists():
        result['stopped_by_request'] = True
        (a.output/'stopped_summary.json').write_text(json.dumps(result, indent=2))
        return
    assert result['additional_macro_steps'] == a.steps
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    rows = evaluate_parallel(a.output, a.output/'parallel_after', workers=2)['trajectories']
    for row in rows:
        name = f"after_{row['case']}.npz"
        shutil.copy2(a.output/'parallel_after'/name, a.output/name)
    result.update(after=rows, evaluation_pending=False)
    (a.output/'after.json').write_text(json.dumps(rows, indent=2))
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    audit(a.output)
    audit_prefix(a.output)
    audit_fixed(a.output)
    print(json.dumps(dict(completed=True, successes=sum(r['success'] for r in rows),
                         macro_steps=result['additional_macro_steps'], teacher_steps=result['teacher_steps'])), flush=True)


if __name__ == '__main__':
    main()


