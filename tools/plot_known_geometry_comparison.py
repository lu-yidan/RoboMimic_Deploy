"""Compare three audited candidates on the same already-observed geometry cases."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.compare_audited_closure_candidates import load_run,sha


def main():
    p=argparse.ArgumentParser()
    for key in ('baseline','trained','refined','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();roles=('baseline','trained','refined');paths=[getattr(a,k) for k in roles]
    runs=[load_run(path) for path in paths]
    for run in runs[1:]:
        for key in ('cases','reference_sha256','controller_modifier'):assert run[key]==runs[0][key]
    assert runs[0]['checkpoint_sha256']==runs[2]['checkpoint_sha256']
    assert len(runs[0]['cases'])==16
    fig,axes=plt.subplots(1,3,figsize=(15,5.8),layout='constrained',sharex=True,sharey=True)
    labels=['Original fixed reference','98,304-step landing PPO','Refined constant reference']
    for ax,label,run in zip(axes,labels,runs):
        for case,offsets in run['cases'].items():
            x,_,z=[1000*v for v in offsets['rung_01']];success=run['rows'][case]['success']
            ax.scatter(x,z,s=70,color='#24876f' if success else '#b75242',marker='o' if success else 'X',zorder=3)
            ax.annotate(case.replace('interior_','I').replace('boundary_','B'),(x,z),xytext=(4,5),textcoords='offset points',fontsize=8)
        count=sum(r['success'] for r in run['rows'].values())
        ax.set_title(f'{label}\n{count}/16 accepted');ax.set_xlabel('Rod x offset (mm)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Rod z offset (mm)')
    fig.suptitle('Sixteen already-observed development geometries\nGreen circle = accepted; red cross = failed. No independent generalization claim.',fontsize=12)
    a.output.mkdir(exist_ok=False);fig.savefig(a.output/'geometry_outcomes.png',dpi=180);fig.savefig(a.output/'geometry_outcomes.pdf');plt.close(fig)
    report=dict(inputs={role:run for role,run in zip(roles,runs)},
                successes={role:sum(r['success'] for r in run['rows'].values()) for role,run in zip(roles,runs)},
                refined_gained_over_baseline=[k for k,r in runs[2]['rows'].items() if r['success'] and not runs[0]['rows'][k]['success']],
                refined_lost_from_baseline=[k for k,r in runs[2]['rows'].items() if not r['success'] and runs[0]['rows'][k]['success']],
                refined_lost_from_trained=[k for k,r in runs[2]['rows'].items() if not r['success'] and runs[1]['rows'][k]['success']],
                source_sha256=sha(__file__),automatic_promotion=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2));(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k!='inputs'}))


if __name__=='__main__':main()
