"""Plot complete audited geometry suites without fitting a success boundary."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    paths = [a.baseline]+([a.candidate] if a.candidate else [])
    runs = [load_run(path) for path in paths]
    protocols = [json.loads((path/'holdout_protocol.json').read_text()) for path in paths]
    assert all(run['cases'] == runs[0]['cases'] for run in runs)
    assert all(protocol == protocols[0] for protocol in protocols)
    fig, axes = plt.subplots(1, len(runs), figsize=(7*len(runs), 6), squeeze=False, layout='constrained', sharex=True, sharey=True)
    for ax, run, label in zip(axes[0], runs, ['Fixed-reference baseline', 'Landing-window PPO']):
        for case, offsets in run['cases'].items():
            x, _, z = np.asarray(offsets['rung_01'])*1000
            success = run['rows'][case]['success']
            color = '#24876f' if success else '#b75242'
            ax.scatter(x, z, s=90, color=color, marker='o' if success else 'X', zorder=3)
            tag = case.replace('interior_', 'I').replace('boundary_', 'B')
            offset = {'B04': (-27, -15), 'B05': (-28, 10), 'B07': (5, 10)}.get(tag, (4, 5))
            ax.annotate(tag, (x, z), xytext=offset, textcoords='offset points', fontsize=8)
        count = sum(row['success'] for row in run['rows'].values())
        ax.set_title(f'{label}: {count}/{len(run["rows"])} accepted')
        ax.set_xlabel('Rod x offset (mm)')
        ax.set_ylabel('Rod z offset (mm)')
        ax.set_xlim(-5.8, 5.8)
        ax.set_ylim(4.5, 16.2)
        ax.grid(alpha=.2)
        ax.axhspan(14, 15.5, color='#777777', alpha=.05)
    legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#24876f', label='Accepted', markersize=9),
              Line2D([0], [0], marker='X', color='w', markerfacecolor='#b75242', label='Failed', markersize=9)]
    axes[0, 0].legend(handles=legend, loc='lower left')
    fig.suptitle('Sixteen predeclared geometries: complete-cycle outcomes\n'
                 'I = interior; B = boundary stratum\n'
                 'Finite test points; no estimated success region or probability.', fontsize=11)
    a.output.mkdir(exist_ok=False)
    fig.savefig(a.output/'geometry_outcomes.png', dpi=180)
    fig.savefig(a.output/'geometry_outcomes.pdf')
    plt.close(fig)
    result = dict(inputs=[dict(directory=run['directory'], audit_sha256=run['audit_sha256']) for run in runs],
                  protocol_sha256=sha(a.baseline/'holdout_protocol.json'), source_sha256=sha(__file__))
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
