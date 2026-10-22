"""Compare completed deterministic grids under the extended position acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    previous = ROOT / 'outputs/contact_frame_continued_grid_v1'
    original = ROOT / 'outputs/clearance_back20_leg14_grid_v1'
    old = {r['name']: r for r in read(previous / 'results.json')}
    new = {r['name']: r for r in read(a.new / 'results.json')}
    base = {r['name']: r for r in read(original / 'results.json')}
    if set(new) != set(old) or len(new) != 17:
        raise ValueError('Comparison requires the complete predeclared 17-case grid')
    rows = []
    for name, after in new.items():
        new_run = a.new / name
        old_run = ROOT / old[name]['source']
        m1 = read(old_run / 'manifest.json'); m2 = read(new_run / 'manifest.json')
        assert m1['plan_sha256'] == m2['plan_sha256'], name
        assert read(old_run / 'scene_variant.json')['geometry_sha256'] == read(new_run / 'scene_variant.json')['geometry_sha256'], name
        for key in ['joint_kp', 'torque_weight', 'leg_kp', 'leg_gain_mode', 'rung_offsets']:
            assert m1['config'][key] == m2['config'][key], (name, key)
        for rel in ['training/climb_cycle.py', 'training/wall_support_env.py', 'training/climb_env.py', 'training/contact_frame_controller.py']:
            assert m1['sources'][rel] == m2['sources'][rel], (name, rel)
        before_pass = bool(old[name]['passed'] and read(old_run / 'joint_limit_audit.json')['passed'])
        base_pass = bool(base[name]['passed'] and read(original / name / 'joint_limit_audit.json')['passed'])
        runtime = read(new_run / 'summary.json')
        independent = read(new_run / 'independent_audit.json')
        native = read(new_run / 'joint_limit_audit.json')
        after_pass = bool(runtime['passed'] and independent['passed'] and native['passed'])
        assert bool(after['passed']) == after_pass, name
        rows.append(dict(case=name, original_passed=base_pass, previous_contact_passed=before_pass, early_contact_passed=after_pass, completed=runtime['completed'], reason=runtime['reason'], failed_checks=[k for k, v in runtime['checks'].items() if not v], minimum_runtime_joint_margin_rad=runtime['minimum_native_joint_margin_rad'], final_joint_error_rad=runtime['final'].get('max_joint_error_rad')))
    result = dict(cases=rows, count=len(rows), original_passes=sum(r['original_passed'] for r in rows), previous_contact_passes=sum(r['previous_contact_passed'] for r in rows), early_contact_passes=sum(r['early_contact_passed'] for r in rows), completed=sum(r['completed'] for r in rows), added=[r['case'] for r in rows if r['early_contact_passed'] and not r['previous_contact_passed']], lost=[r['case'] for r in rows if not r['early_contact_passed'] and r['previous_contact_passed']], scope='17 deterministic cases, not a probability. Same reference/geometry/core/gain parameters; switching timing and missing-contact fallback differ. Low-level contact geometry is privileged. Prior saved-state native-position audit; new runtime per-physics-step and saved-state audit.', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'summary.json').write_text(json.dumps(result, indent=2))
    (a.out / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
