"""Plot audited highest-rung endpoints; silently exclude no incomplete runs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hashlib
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_run(path):
    audit = json.loads((path / 'independent_training_audit.json').read_text())
    manifest = json.loads((path / 'manifest.json').read_text())
    summary = json.loads((path / 'summary.json').read_text())
    assert audit['passed'] and not summary['evaluation_pending']
    for filename, key in [('manifest.json', 'experiment_manifest_sha256'),
                          ('summary.json', 'experiment_summary_sha256'),
                          ('policy.zip', 'checkpoint_sha256')]:
        assert sha(path / filename) == audit[key]
    assert sha(manifest['reference']) == manifest['reference_sha256']
    with np.load(manifest['reference']) as plan:
        end = float(plan['time'][-1])
        initial_z = float(plan['qpos'][0, 2])
    rows = {}
    for record in audit['trajectories']:
        if record['label'] != 'after':
            continue
        trajectory = path / f"after_{record['case']}.npz"
        assert record['passed'] and sha(trajectory) == record['trajectory_sha256']
        endpoint = record['independent_endpoints'][0]
        with np.load(trajectory) as z:
            mask = (z['time_s'] > end - 2.) & (z['time_s'] <= end + 1e-7)
            rise = z['qpos'][mask, 2] - initial_z
        rows[record['case']] = dict(success=record['independently_accepted'],
            joint_closure_mrad=1000 * endpoint['joint_closure_rad'],
            root_rise_max_mm=float(rise.max() * 1000) if len(rise) else None,
            root_rise_min_mm=float(rise.min() * 1000) if len(rise) else None,
            joint_speed_rad_s=endpoint['joint_speed_rad_s'],
            effector_error_mm=1000 * endpoint['effector_error_m'],
            endpoint=endpoint)
    assert set(rows) == set(manifest['evaluation_cases'])
    return rows, manifest, summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-curriculum',action='store_true')
    parser.add_argument('--include-critic',action='store_true')
    args=parser.parse_args()
    experiments=[('Selected','closure_margin_control98304_v1'),
      ('Continue','selected_continuation98304_20261023_v1'),
      ('Quality reward','closure_quality98304_20261024_v1'),
      ('Leg exploration','closure_exploration98304_20261024_v1')]
    if args.include_curriculum:experiments.append(('Suffix curriculum','closure_curriculum98304_20261024_v2'))
    if args.include_critic:experiments.append(('Calibrated critic','closure_curriculum_critic98304_20261024_v1'))
    rows=[]
    for label,name in experiments:
        path=Path('outputs')/name
        result,manifest,summary=read_run(path)
        rows.append(dict(label=label,directory=str(path),highest=result['foot_z_plus15'],
          five_case_successes=sum(bool(v['success']) for v in result.values()),
          audit_sha256=sha(path/'independent_training_audit.json')))
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    xs=list(range(len(rows)))
    for ax,key,threshold,title,unit in [
        (axes[0],'joint_closure_mrad',40.,'Joint closure: last 2 s maximum','mrad'),
        (axes[1],'root_rise_max_mm',260.,'Root rise: last 2 s maximum','mm')]:
        values=[r['highest'][key] for r in rows]
        ax.plot(xs,values,'o-',color='#215B87',lw=2)
        ax.axhline(threshold,color='#B84735',ls='--',label='Upper task limit')
        ax.set_xticks(xs,[r['label'] for r in rows],rotation=20,ha='right')
        ax.set_ylabel(unit);ax.set_title(title);ax.grid(axis='y',alpha=.2)
        for x,y in zip(xs,values):ax.annotate(f'{y:.3f}',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
        ax.margins(y=.25);ax.legend(loc='lower left')
    fig.suptitle('Known +15 mm case | full-cycle deterministic evaluation | original gates',fontsize=12)
    args.output.mkdir(exist_ok=False)
    fig.savefig(args.output/'highest_rung.png',dpi=180)
    fig.savefig(args.output/'highest_rung.pdf')
    plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(dict(rows=rows,source_sha256=sha(__file__),scope='Known cases used for development, not held-out generalization; endpoints alone do not replace all success gates.'),indent=2))
    (args.output/'plot_source.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
