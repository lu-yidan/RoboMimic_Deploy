"""Report all predeclared fresh geometries for three frozen controllers."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    parser = argparse.ArgumentParser()
    for key in ('baseline', 'trained', 'refined', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    roles = ['baseline', 'trained', 'refined']
    paths = [getattr(args, key) for key in roles]
    runs = [load_run(path) for path in paths]
    protocol = json.loads((paths[0]/'holdout_protocol.json').read_text())
    protocol_hash = sha(paths[0]/'holdout_protocol.json')
    results = []
    for role, path, run in zip(roles, paths, runs):
        manifest = json.loads((path/'manifest.json').read_text())
        result = json.loads((path/'geometry_result.json').read_text())
        assert sha(path/'holdout_protocol.json') == protocol_hash
        assert result['protocol_sha256'] == protocol_hash
        assert manifest['fresh_role'] == role
        assert run['cases'] == protocol['cases']
        assert run['checkpoint_sha256'] == protocol['candidates'][role]['checkpoint_sha256']
        assert run['adaptive_reference_command'] == protocol['candidates'][role]['command']
        assert result['successes'] == sum(row['success'] for row in run['rows'].values())
        results.append(result)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), layout='constrained', sharex=True, sharey=True)
    labels = ['Original fixed reference', '98,304-step landing PPO', 'Refined constant reference']
    for ax, label, run, result in zip(axes, labels, runs, results):
        for case, offsets in run['cases'].items():
            x, _, z = [1000*v for v in offsets['rung_01']]
            success = run['rows'][case]['success']
            ax.scatter(x, z, s=80, color='#24876f' if success else '#b75242',
                       marker='o' if success else 'X', zorder=3)
            tag = case.replace('fresh_interior_', 'I').replace('fresh_boundary_', 'B')
            ax.annotate(tag, (x, z), xytext=(4, 5), textcoords='offset points', fontsize=8)
        ax.set_title(f"{label}\n{result['successes']}/{result['cases']} accepted")
        ax.set_xlabel('Rod x offset (mm)')
        ax.set_xlim(-5.8, 5.8)
        ax.set_ylim(4.5, 16.2)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Rod z offset (mm)')
    fig.suptitle('Eight fresh geometries; controller identities fixed before rollout\n'
                 'Green circle = accepted; red cross = failed; I = interior; B = boundary\n'
                 'Finite paired tests, not an estimated success probability.', fontsize=11)
    args.output.mkdir(exist_ok=False)
    fig.savefig(args.output/'geometry_outcomes.png', dpi=180)
    fig.savefig(args.output/'geometry_outcomes.pdf')
    plt.close(fig)
    report = dict(protocol_sha256=protocol_hash, source_sha256=sha(__file__),
                  inputs=[dict(role=role, run=run, result=result) for role, run, result in zip(roles, runs, results)],
                  automatic_promotion=False)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2))
    (args.output/'report_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({role: result for role, result in zip(roles, results)}))


if __name__ == '__main__':
    main()
