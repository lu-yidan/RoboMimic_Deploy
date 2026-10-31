"""Plot audited development margins for identical controller authority."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    paths=[Path('outputs/closure_fivecase_eightcase_comparison_20261024_v1/comparison.json'),Path('outputs/closure_five_vs_eight_training_20261024_v1/comparison.json')]
    first,second=[json.loads(p.read_text()) for p in paths]
    assert first['after']['checkpoint_sha256']==second['before']['checkpoint_sha256']
    models=[('Initial search seed',first['before'],'#287a8b'),('PPO: 5 training cases',first['after'],'#bd6744'),('PPO: 8 training cases',second['after'],'#69874c')]
    for _,run,_ in models:
        folder=Path(run['directory']);assert digest(folder/'policy.zip')==run['checkpoint_sha256']
        audit=folder/('independent_candidate_audit.json' if (folder/'independent_candidate_audit.json').exists() else 'independent_training_audit.json')
        assert digest(audit)==run['audit_sha256']
        assert sum(r['success'] for r in run['rows'].values())==5
    cases=list(first['before']['cases']);labels=['z15','z12.5','z10','z5','z7.5','z15.1','x-0.1/z15','x+0.1/z15']
    hard=['foot_z_plus15','z15.1','x-0.1_z15','x+0.1_z15']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for index,(label,run,color) in enumerate(models):
        axes[0].scatter(np.arange(8)+(index-1)*.18,[40-run['rows'][c]['closure_mrad'] for c in cases],label=label,color=color,s=35)
        axes[1].scatter(np.arange(4)+(index-1)*.18,[260-run['rows'][c]['rise_max_mm'] for c in hard],label=label,color=color,s=35)
    axes[0].set_xticks(np.arange(8),labels,rotation=35,ha='right');axes[0].set_ylabel('Joint closure margin (mrad)')
    axes[1].set_xticks(np.arange(4),['z15','z15.1','x-0.1/z15','x+0.1/z15'],rotation=20,ha='right');axes[1].set_ylabel('Upper root-rise margin (mm)')
    axes[0].set_title('All eight cases');axes[1].set_title('Four difficult cases: very small height margins')
    for axis in axes:
        axis.axhline(0,color='#333333',ls='--',lw=1);axis.grid(axis='y',alpha=.2);axis.set_xlabel('Rod offsets (mm)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Same controller, equal PPO budget: all three candidates remain 5/8\nSelected endpoint margins only; full acceptance requires every original task check',fontsize=11)
    args.output.mkdir(exist_ok=False)
    for suffix in ('png','pdf'):fig.savefig(args.output/f'training_comparison.{suffix}',dpi=180)
    (args.output/'manifest.json').write_text(json.dumps(dict(inputs={str(p):digest(p) for p in paths},source_sha256=digest(__file__),scope='Deterministic development results, one training seed. No confidence interval or generalization claim.'),indent=2))
    (args.output/'plot_source.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
