"""Extend the unchanged two-cycle bridge with nominal periodic references."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.audit_multirung_cycle import sha
from tools.plan_climb_cycle import audit_plan
from tools.repeat_climb_cycle import repeat_reference
from training.wall_support_env import WallSupportEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bridge', type=Path, required=True)
    p.add_argument('--nominal', type=Path, required=True)
    p.add_argument('--cycles', type=int, choices=range(3, 9), default=8)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    for path in (a.bridge, a.nominal):
        assert sha(path) == json.loads((path.parent/'manifest.json').read_text())['plan_sha256']
        assert json.loads((path.parent/'audit.json').read_text())['passed']
    bridge = dict(np.load(a.bridge))
    nominal = dict(np.load(a.nominal))
    assert len(bridge['cycle_ends_s']) == 2
    repeated = repeat_reference(nominal, a.cycles)
    boundary = float(bridge['time'][-1])
    assert abs(boundary - repeated['cycle_ends_s'][1]) < 1e-10
    index = int(np.argmin(abs(repeated['time']-boundary)))
    assert abs(repeated['time'][index]-boundary) < 1e-10
    assert np.allclose(bridge['qpos'][-1], repeated['qpos'][index], atol=1e-10, rtol=0)
    keys = ('qpos', 'time', 'phase', 'active', 'errors')
    plan = {key: np.concatenate([bridge[key], repeated[key][index+1:]]) for key in keys}
    plan.update(initial=bridge['initial'], cycle_ends_s=repeated['cycle_ends_s'], cycle_rise_m=np.array(.25))
    exact = {key: bool(np.array_equal(plan[key][:len(bridge[key])], bridge[key])) for key in keys}
    assert all(exact.values()) and np.all(np.diff(plan['time']) > 0)
    e = WallSupportEnv(root/'outputs/wall_support_bundle_pd15_v2')
    try:
        e.reset()
        audit = audit_plan(e, plan)
        audit['inherited_parent_ik_residual'] = audit.pop('ik_residual')
        audit['scope'] = 'Dense nominal-scene kinematic screening only; no new IK certificate or dynamics acceptance.'
        a.output.mkdir(exist_ok=False)
        np.savez_compressed(a.output/'plan.npz', **plan)
        (a.output/'audit.json').write_text(json.dumps(audit, indent=2))
        parent = json.loads((a.bridge.parent/'manifest.json').read_text())
        manifest = dict(plan_sha256=sha(a.output/'plan.npz'), bridge_reference=str(a.bridge.resolve()),
            bridge_sha256=sha(a.bridge), nominal_reference=str(a.nominal.resolve()), nominal_sha256=sha(a.nominal),
            first_cycle_exact=parent['first_cycle_exact'], first_two_cycles_exact=exact,
            blend_seconds=parent['blend_seconds'], cycles=a.cycles, horizon_s=float(plan['time'][-1]),
            source_sha256=sha(__file__), repeat_source_sha256=sha('tools/repeat_climb_cycle.py'),
            scope='First two reference cycles unchanged; subsequent nominal cycles. Requires continuous dynamics validation.')
        (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
        (a.output/'planner_source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps(dict(passed=audit['passed'], failures=len(audit['failures']), **manifest)), flush=True)
    finally:
        e.close()


if __name__ == '__main__':
    main()
