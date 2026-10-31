"""Compare terminal exploration to its completed reward-only control."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha
from tools.report_selected_continuation import read_run


def main():
    candidate = Path('outputs/closure_exploration98304_20261024_v1')
    protocol = json.loads((candidate / 'matched_protocol.json').read_text())
    control = Path(protocol['control_reuse'])
    for name, digest in protocol['control_reuse_hashes'].items():
        assert sha(control / name) == digest
    before, bm, bs = read_run(control)
    after, am, ats = read_run(candidate)
    for key in ('parent_checkpoint_sha256', 'parent_steps', 'seed', 'training_cases',
                'evaluation_cases', 'reference_sha256', 'config', 'environments'):
        assert bm[key] == am[key], key
    assert bs['additional_steps'] == ats['additional_steps'] == protocol['steps']
    assert am['initial_policy_max_difference'] == am['initial_optimizer_max_difference'] == 0.
    rows = [dict(case=k, control=before[k], exploration=after[k]) for k in before]
    gained = [k for k in before if after[k]['success'] and not before[k]['success']]
    lost = [k for k in before if before[k]['success'] and not after[k]['success']]
    result = dict(rows=rows, gained=gained, lost=lost,
        eligible_for_grid=bool(gained and not lost),
        training_additional_steps=ats['additional_steps'],
        training_bonus_sum=ats['total_training_quality_bonus'],
        integrity_audit_passed=True,
        control_audit_sha256=sha(control / 'independent_training_audit.json'),
        candidate_audit_sha256=sha(candidate / 'independent_training_audit.json'),
        scope=protocol['scope'])
    out = Path('outputs/closure_exploration_comparison_20261024_v1')
    out.mkdir(exist_ok=False)
    (out / 'comparison.json').write_text(json.dumps(result, indent=2))
    (out / 'report_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
