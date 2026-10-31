"""Prepare a continuous two-cycle reference; no simulator state is overwritten."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.wall_support_env import WallSupportEnv
from tools.plan_climb_cycle import audit_plan
from tools.continue_vector_full_cycle_training import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--first', type=Path, required=True)
    p.add_argument('--next', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--blend-seconds', type=float, default=6.)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert 1. <= a.blend_seconds <= 10.
    for source in (a.first, a.next):
        manifest = json.loads((source.parent/'manifest.json').read_text())
        assert manifest['plan_sha256'] == sha(source)
        assert json.loads((source.parent/'audit.json').read_text())['passed']
    with np.load(a.first) as z:
        first = {k: z[k].copy() for k in z.files}
    with np.load(a.next) as z:
        second = {k: z[k].copy() for k in z.files}
    assert np.array_equal(first['qpos'][0], second['qpos'][0])
    assert first['time'][0] == second['time'][0] == 0
    assert len(first.get('cycle_ends_s', [0])) == len(second.get('cycle_ends_s', [0])) == 1
    shift = np.array([.25*np.tan(np.deg2rad(5)), 0., .25])
    assert np.allclose(second['qpos'][-1, :3]-second['qpos'][0, :3], shift, atol=1e-7, rtol=0)
    assert np.max(abs(second['qpos'][-1, 3:]-second['qpos'][0, 3:])) < 1e-4
    shifted = second['qpos'].copy()
    shifted[:, :3] += shift
    delta = first['qpos'][-1]-shifted[0]
    u = np.clip(second['time']/a.blend_seconds, 0., 1.)
    weight = 1.-(10*u**3-15*u**4+6*u**5)
    shifted += weight[:, None]*delta
    shifted[0] = first['qpos'][-1].copy()
    assert np.max(abs(np.linalg.norm(shifted[:, 3:7], axis=1)-1.)) < 1e-10
    plan = {key: np.concatenate([first[key], (shifted if key == 'qpos' else
            second[key]+first['time'][-1] if key == 'time' else second[key])[1:]])
            for key in ('qpos', 'time', 'phase', 'active', 'errors')}
    plan.update(initial=first['qpos'][0].copy(),
                cycle_ends_s=np.array([first['time'][-1], first['time'][-1]+second['time'][-1]]),
                cycle_rise_m=np.array(.25))
    prefix = {key: bool(np.array_equal(plan[key][:len(first[key])], first[key]))
              for key in ('qpos', 'time', 'phase', 'active', 'errors')}
    assert all(prefix.values()) and np.all(np.diff(plan['time']) > 0)
    env = WallSupportEnv(root/'outputs/wall_support_bundle_pd15_v2')
    try:
        env.reset()
        audit = audit_plan(env, plan)
        audit['inherited_parent_ik_residual'] = audit.pop('ik_residual')
        audit['scope'] = ('Nominal-scene dense collision and native-range screening only. '
            'Blended second-cycle targets have no new contact-pose IK certificate. '
            'A passing kinematic screen does not establish support, torque feasibility or dynamic completion.')
        a.output.mkdir(exist_ok=False)
        np.savez_compressed(a.output/'plan.npz', **plan)
        (a.output/'audit.json').write_text(json.dumps(audit, indent=2))
        sources = {name: sha(name) for name in (str(Path(__file__)), 'tools/plan_climb_cycle.py')}
        manifest = dict(first_reference=str(a.first.resolve()), first_reference_sha256=sha(a.first),
            next_reference=str(a.next.resolve()), next_reference_sha256=sha(a.next),
            plan_sha256=sha(a.output/'plan.npz'), bundle_manifest_sha256=sha(env.bundle/'manifest.json'),
            nominal_shift_m=shift.tolist(), second_cycle_bridge_delta=delta.tolist(),
            blend_seconds=a.blend_seconds, blend='quintic smoothstep decay of reference-pose discrepancy',
            first_cycle_exact=prefix, sources=sources,
            scope='New two-cycle reference only. First 113.66-second reference is unchanged. '
                  'Second reference starts continuously and returns to the periodic nominal trajectory. '
                  'No change to physical state, existing experiments, native limits or endpoint acceptance.')
        (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
        (a.output/'planner_source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps(dict(passed=audit['passed'], failures=len(audit['failures']),
            joint_margin_rad=audit['joint_margin_rad'], maximum_penetration_m=audit['maximum_penetration_m'],
            horizon_s=float(plan['time'][-1]), first_cycle_exact=prefix,
            bridge_position_delta_m=delta[:3].tolist())), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
