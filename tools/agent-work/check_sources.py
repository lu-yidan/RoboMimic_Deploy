"""Verify an experiment directory against its manifest: was anything changed after the run?

Prints two groups, mirroring what the project's own audits enforce:
  * training/  -> enforced by tools/audit_action_bias_candidate_v2.py (current_environment_matches_frozen)
  * tools/     -> recorded for provenance but not enforced (tool files may evolve between runs)

Usage:  python check_sources.py <experiment-dir>
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8-sig'))
    sources = manifest['sources']
    groups = {'training': [], 'tools': []}
    for rel, digest in sources.items():
        key = rel.replace('\\', '/')
        group = 'training' if key.startswith('training/') else ('tools' if key.startswith('tools/') else None)
        if group is None:
            continue
        path = REPO / rel
        if not path.exists():
            groups[group].append((rel, 'MISSING'))
        elif sha(path) != digest:
            groups[group].append((rel, 'CHANGED'))

    print(f'run: {run.name}')
    print(f'recorded sources: {len(sources)} (training {sum(1 for r in sources if r.startswith("training"))}, '
          f'tools {sum(1 for r in sources if r.startswith("tools"))})')
    print()
    print('[enforced] training/ mismatches:', len(groups['training']))
    for rel, state in groups['training']:
        print('   ', state, rel)
    print('[provenance only] tools/ mismatches:', len(groups['tools']))
    for rel, state in groups['tools']:
        print('   ', state, rel)
    print()
    print('verdict:', 'reproducible (training sources unchanged)' if not groups['training']
          else 'training sources changed -> the audited chain no longer matches this run')
    print('note: tools/ changes do not break reproducibility; the run keeps its own source_snapshot '
          'and the audits only enforce training/*.')
    return 0 if not groups['training'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
