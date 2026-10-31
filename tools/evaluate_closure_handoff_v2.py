"""Continue an accepted high-rod cycle into a nominal second cycle without resets."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.bar_geometry_observation import BarGeometryObservation, OBSERVATION_CONTRACT
from training.closure_composite_policy import ClosureCompositePolicy
from training.closure_handoff_reference_control import (
    install_waist_target, install_body_target, install_adaptive_reference,
    body_target_offset, handoff_fade)
from tools.compare_audited_closure_candidates import load_run, sha
from tools.audit_multirung_cycle import audit_endpoints


def main():
    p = argparse.ArgumentParser()
    for name in ('candidate', 'plan', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--case', default='foot_z_plus15')
    a = p.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    verified = load_run(a.candidate)
    m = json.loads((a.candidate/'manifest.json').read_text())
    pm = json.loads((a.plan.parent/'manifest.json').read_text())
    assert m['observation_contract'] == OBSERVATION_CONTRACT
    assert verified['rows'][a.case]['success'] and verified['rows'][a.case]['complete']
    assert pm['first_reference_sha256'] == m['reference_sha256']
    assert pm['plan_sha256'] == sha(a.plan) and all(pm['first_cycle_exact'].values())
    for source, digest in m['sources'].items():
        assert sha(a.candidate/'source_snapshot'/source) == digest
        if source.replace('\\', '/').startswith('training/'):
            assert sha(root/source) == digest
    env = FullCycleResidualEnv(plan=a.plan, cases={a.case: m['evaluation_cases'][a.case]},
        normal_bias_y_deg=m['normal_bias_y_deg'], memory_enabled=m['force_history_memory'])
    assert len(env.ends) == 2 and float(env.ends[0]) == m['horizon_seconds']
    first_end = float(env.ends[0])
    boundary = round(first_end/.02)
    observer = BarGeometryObservation(env, mode=m['geometry_mode'])
    model = ClosureCompositePolicy(PPO.load(a.candidate/'policy.zip', device='cpu'), m)
    raw, _ = observer.reset(seed=20261022, options={'case': a.case})
    c = m['controller_modifier']
    gain = float(np.clip((float(raw[138])-.625)/.125, 0., 1.))
    fade_seconds = pm['blend_seconds']
    waist = install_waist_target(env, c['waist_bias'], geometry_gain=gain,
        ramp_seconds=c['waist_ramp_seconds'], record=True, fade_seconds=fade_seconds)
    drop = gain*c['body_drop_m']
    install_body_target(env, drop, fade_seconds=fade_seconds)
    extra = install_adaptive_reference(env, c, geometry_gain=gain, record=True, fade_seconds=fade_seconds)
    extra['set_action'](m['adaptive_reference_command'])
    with np.load(a.candidate/f'after_{a.case}.npz') as z:
        original = {key: z[key].copy() for key in z.files}
    a.output.mkdir(exist_ok=False)
    shutil.copy2(a.plan, a.output/'reference.npz')
    sources = dict(m['sources'])
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if not name:
            continue
        source = Path(name).resolve()
        if source.suffix == '.py' and source.is_relative_to(root):
            relative = source.relative_to(root)
            if relative.parts[0] in ('tools', 'training'):
                sources[str(relative)] = sha(source)
    for source, digest in sources.items():
        assert sha(root/source) == digest
        destination = a.output/'source_snapshot'/source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root/source).read_bytes())
    manifest = dict(bundle=str(env.env.bundle), sources=sources, plan_sha256=sha(a.plan),
        parent_experiment=str(a.candidate.resolve()), parent_audit_sha256=verified['audit_sha256'],
        checkpoint_sha256=verified['checkpoint_sha256'], case=a.case, offsets=m['evaluation_cases'][a.case],
        reference_manifest_sha256=sha(a.plan.parent/'manifest.json'), fade_seconds=fade_seconds,
        scope='First cycle uses the audited composite actor and fixed references. Second cycle uses classical control '
              'after a six-second smooth decay of the carried last actor action and reference offsets. '
              'Physical state and controller/filter history never reset at the seam. Original strict endpoint '
              'and native physical criteria remain active at both boundaries. Not a learned two-cycle policy.')
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (a.output/'scene_variant.json').write_text(json.dumps(env.variant, indent=2))
    records = dict(qpos=[env.env.data.qpos.copy()], qvel=[env.env.data.qvel.copy()], time_s=[0.],
                   observations=[], raw_observations=[], actions=[], torques=[], rewards=[])
    carried = None
    prefix = None
    started = time.perf_counter()
    try:
        while True:
            t = float(env.env.data.time)
            index = min(int(np.searchsorted(env.ends, t, side='right')), len(env.ends)-1)
            start = 0. if index == 0 else float(env.ends[index-1])
            obs = raw.copy()
            obs[105] = max(0., 1.-(t-start)/(float(env.ends[index])-start))
            if len(records['actions']) < boundary:
                action = model.predict(obs, deterministic=True)[0]
                carried = action.copy()
            else:
                action = (carried*handoff_fade(t, first_end, fade_seconds)[0]).astype(np.float32)
            records['observations'].append(obs.copy())
            records['raw_observations'].append(raw.copy())
            raw, reward, done, truncated, info = observer.step(action)
            records['actions'].append(action.copy())
            records['torques'].append(env.last_torque.copy())
            records['rewards'].append(float(reward))
            records['qpos'].append(env.env.data.qpos.copy())
            records['qvel'].append(env.env.data.qvel.copy())
            records['time_s'].append(float(env.env.data.time))
            if len(records['actions']) == boundary:
                prefix = {key: bool(np.array_equal(np.asarray(records[key]), original[key]))
                          for key in ('qpos', 'qvel', 'time_s', 'observations', 'actions', 'torques')}
                prefix['waist_reference_trace'] = bool(np.array_equal(np.asarray(waist['trace']).reshape(-1, 10), original['waist_reference_trace']))
                prefix['adaptive_reference_trace'] = bool(np.array_equal(np.asarray(extra['trace']).reshape(-1, 21), original['adaptive_reference_trace']))
                prefix['rewards_before_terminal_bonus'] = bool(np.array_equal(np.asarray(records['rewards'])[:-1], original['rewards'][:-1]))
                prefix['terminal_bonus_only_difference'] = abs(records['rewards'][-1]-(float(original['rewards'][-1])-20.)) < 1e-9
                (a.output/'first_cycle_prefix.json').write_text(json.dumps(dict(passed=all(prefix.values()), checks=prefix), indent=2))
                print(json.dumps(dict(first_cycle_prefix_exact=all(prefix.values()), checks=prefix)), flush=True)
                if not all(prefix.values()):
                    info.update(success=False, reason='first_cycle_prefix_mismatch')
                    break
            if len(records['actions']) % 1000 == 0:
                print(json.dumps(dict(time_s=info['time_s'], steps=len(records['actions']))), flush=True)
            if done or truncated:
                break
        arrays = {key: np.asarray(value) for key, value in records.items()}
        arrays.update(waist_reference_trace=np.asarray(waist['trace']).reshape(-1, 10),
            adaptive_reference_trace=np.asarray(extra['trace']).reshape(-1, 21),
            body_reference_trace=np.asarray([body_target_offset(t, drop, first_end=first_end, fade_seconds=fade_seconds) for t in records['time_s']]))
        trajectory = a.output/'trajectory.npz'
        np.savez_compressed(trajectory, **arrays)
        (a.output/'runtime_info.json').write_text(json.dumps(info, indent=2))
        endpoints = audit_endpoints(env.env, arrays, env.plan)
        result = dict(info, first_cycle_prefix_exact=bool(prefix and all(prefix.values())),
            prefix_checks=prefix, independent_endpoints=endpoints,
            passed=bool(info['success'] and prefix and all(prefix.values()) and all(row['passed'] for row in endpoints)),
            wall_seconds=time.perf_counter()-started, trajectory_sha256=sha(trajectory),
            final_rise_m=float(arrays['qpos'][-1, 2]-arrays['qpos'][0, 2]),
            scope=manifest['scope'])
        (a.output/'summary.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(passed=result['passed'], reason=result['reason'], time_s=result['time_s'],
                             first_cycle_prefix_exact=result['first_cycle_prefix_exact'], final_rise_m=result['final_rise_m'])), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
