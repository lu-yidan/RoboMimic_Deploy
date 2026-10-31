"""Read the finite reward experiment status without touching running jobs."""
import argparse
import datetime
import json
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    state = read(args.root/'status.json') or {}
    result = dict(time=datetime.datetime.now().isoformat(), completed=state.get('completed', False),
                  active_step=state.get('active_step'), arms={})
    for mode in ['average', 'bottleneck']:
        root = args.root/mode
        if not root.exists():
            result['arms'][mode] = dict(started=False)
            continue
        progress = read(root/'training_status.json') or {}
        summary = read(root/'summary.json') or {}
        arm = dict(started=True, additional_steps=summary.get('additional_steps', progress.get('additional_steps', 0)),
                   completed_episodes=summary.get('completed_training_episodes', progress.get('completed_episodes', 0)))
        episodes = root/'training_episodes.jsonl'
        if episodes.exists():
            rows = []
            for line in episodes.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            arm['logged_successful_training_episodes'] = sum(row['success'] for row in rows)
            arm['completed_episodes'] = len(rows)
        evaluation = root/'parallel_after'
        cases = [read(path) for path in evaluation.glob('*/result.json')]
        cases = [case for case in cases if case is not None]
        arm['development16_recorded'] = len(cases)
        arm['development16_successes_so_far'] = sum(case['success'] for case in cases)
        arm['independent_audit_passed'] = (read(root/'independent_candidate_audit.json') or {}).get('passed', False)
        result['arms'][mode] = arm
    result['regressions'] = {}
    for mode in ['average', 'bottleneck']:
        for suite, total in [('original8', 8), ('grid12', 12)]:
            label = mode+'_'+suite
            root = args.root/label
            if not root.exists():
                continue
            rows = [read(path) for path in root.glob('parallel_after/*/result.json')]
            rows = [row for row in rows if row is not None]
            result['regressions'][label] = dict(recorded=len(rows), total=total,
                successes_so_far=sum(row['success'] for row in rows), final=read(root/'regression.json'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
