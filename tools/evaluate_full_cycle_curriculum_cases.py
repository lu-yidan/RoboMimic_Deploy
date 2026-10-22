"""Diagnose declared candidate training cases; these are not blind held-out tests."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

from tools.train_full_cycle_residual import evaluate
from training.full_cycle_residual_env import ROOT, CONTRACT

CASES = {
    'foot_z_minus5': {'rung_01': [0., 0., -.005]},
    'foot_z_minus10': {'rung_01': [0., 0., -.01]},
    'foot_z_plus10': {'rung_01': [0., 0., .01]},
    'hand_x_plus5': {'rung_05': [.005, 0., 0.]},
}


def run_case(args):
    name, offsets, root = args
    out = Path(root)/name
    out.mkdir()
    return evaluate(None, out, 'zero', {name: offsets})[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('Use 1..4 workers')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for rel in ('training/full_cycle_residual_env.py', 'tools/train_full_cycle_residual.py',
                'tools/evaluate_full_cycle_curriculum_cases.py'):
        target = args.output/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, target)
        sources[rel] = hashlib.sha256(target.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(dict(contract=CONTRACT, cases=CASES, sources=sources,
        purpose='Candidate curriculum diagnostics; previously investigated geometries, not blind held-out evaluation',
        normal_bias_y_deg=-3., memory_enabled=True), indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_case, [(name, offsets, str(args.output)) for name, offsets in CASES.items()]))
    (args.output/'summary.json').write_text(json.dumps(results, indent=2))
