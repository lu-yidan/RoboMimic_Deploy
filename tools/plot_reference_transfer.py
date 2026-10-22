"""Plot audited reference-transfer outcomes without pooling different targets."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args();p=args.experiment
    data=json.loads((p/'independent_comparison.json').read_text());rows={r['label']:r for r in data['results']}
    for r in rows.values():
        assert r['integrity_passed'] and sha(p/r['label']/'independent_audit.json')==r['audit_sha256']
    order=[f'{name}_bias{bias}' for name in ('original','clearance','adapted') for bias in ('+0','-3')]
    labels=['Original\n0 deg','Original\n-3 deg','Clearance\n0 deg','Clearance\n-3 deg','Adapted\n0 deg','Adapted\n-3 deg']
    fig,axes=plt.subplots(2,1,figsize=(9.5,7.0),layout='constrained')
    colors=['#26835b' if rows[k]['accepted'] else '#b34c43' for k in order]
    times=[rows[k]['time_s'] for k in order];x=np.arange(len(order))
    axes[0].bar(x,times,color=colors,width=.65)
    for i,k in enumerate(order):
        r=rows[k];tag='PASS' if r['accepted'] else ('contact failure' if 'contact' in r['reason'] else 'endpoint failure')
        axes[0].text(i,times[i]+2,f'{times[i]:.2f} s\n{tag}',ha='center',va='bottom',fontsize=8)
    axes[0].set(ylim=(0,140),ylabel='Executed simulation time (s)',xticks=x,xticklabels=labels)
    axes[0].set_title('Known +10 mm foot rung: zero-action reference transfer',loc='left')
    for i,k in enumerate(order):
        ep=rows[k]['endpoints'][0]
        if ep['joint_closure_rad'] is None:
            axes[1].text(i,.005,'No endpoint\nreached',ha='center',fontsize=8)
        else:
            axes[1].bar(i,ep['joint_closure_rad'],color=colors[i],width=.65)
            axes[1].text(i,ep['joint_closure_rad']+.002,f'{ep["joint_closure_rad"]:.4f}',ha='center',fontsize=8)
    axes[1].axhline(.04,color='#273447',ls='--',lw=1,label='Unchanged closure threshold: 0.04 rad')
    axes[1].set(ylim=(0,.092),ylabel='Final 2 s peak joint closure error (rad)',xticks=x,xticklabels=labels)
    axes[1].legend(loc='upper left',fontsize=8)
    for ax in axes:
        ax.set_xlim(-.65,5.65)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.supxlabel('Single deterministic case per condition. Adapted reference has a different endpoint target.\nGreen requires ALL strict checks, not closure alone. No learned-policy or generalization claim.',fontsize=9)
    for extension in ('png','pdf'):fig.savefig(p/f'reference_transfer.{extension}',dpi=170)
    plt.close(fig)
    (p/'plot_provenance.json').write_text(json.dumps(dict(comparison_sha256=sha(p/'independent_comparison.json'),source_sha256=sha(__file__),png_sha256=sha(p/'reference_transfer.png'),pdf_sha256=sha(p/'reference_transfer.pdf')),indent=2))
    print(str(p/'reference_transfer.png'))


if __name__=='__main__':main()
