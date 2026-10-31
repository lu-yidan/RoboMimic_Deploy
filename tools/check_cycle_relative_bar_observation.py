"""Offline interface checks of cycle-relative privileged observations."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.bar_geometry_observation import BarGeometryObservation
from training.cycle_relative_bar_observation import CycleRelativeBarObservation
from tools.audit_multirung_cycle import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    offsets = {'rung_01': [0, 0, .015], 'rung_02': [-.004, 0, 0],
               'rung_06': [0, .002, 0], 'rung_07': [0, 0, .003]}
    env = FullCycleResidualEnv(plan=a.plan, cases={'mapping': {}}, normal_bias_y_deg=0., memory_enabled=True)
    try:
        old = BarGeometryObservation(env, mode='oracle')
        new = CycleRelativeBarObservation(env, mode='oracle')
        blind = CycleRelativeBarObservation(env, mode='blind')
        env.reset(options={'case': 'mapping'})
        # Interface test only: inject site offsets directly, never step physics.
        # The legacy scene-variant helper intentionally supports fewer row IDs.
        for row, offset in offsets.items():
            site = env.env.model.site(row+'_target').id
            env.env.model.site_pos[site] = env.nominal_geometry[1][site]+np.asarray(offset)
        raw = np.linspace(-.4, .4, 136, dtype=np.float32)
        checks = {}
        for t in (0., 99.66, 113.659999999):
            env.env.data.time = t
            checks[f'first_cycle_exact_{t}'] = bool(np.array_equal(new.observation(raw), old.observation(raw)))
        expected = np.array([-.2, 0, 0, 0, 0, 0, 0, .1, 0, 0, 0, .15], dtype=np.float32)
        for t in (113.66, 113.660000001, 227.32):
            env.env.data.time = t
            result = new.observation(raw)
            checks[f'second_roles_{t}'] = bool(np.allclose(result[136:], expected, atol=1e-7, rtol=0))
            checks[f'base_unchanged_{t}'] = bool(np.array_equal(result[:136], raw))
            checks[f'blind_zero_{t}'] = bool(np.array_equal(blind.observation(raw)[136:], np.zeros(12)))
        site = env.env.model.site('rung_02_target').id
        env.env.model.site_pos[site, 0] = env.nominal_geometry[1][site, 0]+.021
        try:
            new.observation(raw)
            checks['reject_21mm'] = False
        except ValueError:
            checks['reject_21mm'] = True
        result = dict(passed=all(checks.values()), checks=checks, role_rows=new.role_rows,
            plan_sha256=sha(a.plan), sources={name: sha(name) for name in
                (str(Path(__file__)), 'training/cycle_relative_bar_observation.py')},
            scope='Offline observation-only mapping and boundary checks; synthetic site offsets and timestamps, no physics rollout or performance claim.')
        a.output.mkdir(exist_ok=False)
        (a.output/'summary.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        assert result['passed']
    finally:
        env.close()


if __name__ == '__main__':
    main()
