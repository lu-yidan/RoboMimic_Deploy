"""Map the already-evaluated local development cases without changing selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run=load_run(a.run)
    assert len(run['rows'])==8
    offsets=np.array([v['rung_01'] for v in run['cases'].values()]);center=offsets.mean(axis=0)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');rows=[]
    styles={'success':('#168474','o'),'height':('#bd532f','s'),'joint':('#7654a3','^'),'other':('#333333','x')}
    for index,(case,row) in enumerate(run['rows'].items()):
        category='success' if row['success'] else ('height' if row['rise_max_mm']>260 else ('joint' if row['closure_mrad']>=40 else 'other'))
        color,marker=styles[category];dx,dz=(np.array(run['cases'][case]['rung_01'])-center)[[0,2]]*1000
        axes[0].scatter(dx,dz,c=color,marker=marker,s=90)
        axes[0].annotate(case.replace('local_',''),(dx,dz),xytext=(7,5),textcoords='offset points')
        joint_margin=40-row['closure_mrad'];height_margin=260-row['rise_max_mm']
        axes[1].scatter(joint_margin,height_margin,c=color,marker=marker,s=70)
        axes[1].annotate(case.replace('local_',''),(joint_margin,height_margin),xytext=(6,4),textcoords='offset points')
        rows.append(dict(case=case,category=category,joint_margin_mrad=joint_margin,height_margin_mm=height_margin,dx_mm=dx,dz_mm=dz))
    for label,(color,marker) in styles.items():
        if any(r['category']==label for r in rows):axes[0].scatter([],[],c=color,marker=marker,label=label)
    axes[0].set(xlabel='Rod x change from B07 center (mm)',ylabel='Rod z change from B07 center (mm)',title='Local development geometry: 8 measured points')
    axes[0].legend(loc='lower left',fontsize=9);axes[0].margins(.2)
    axes[1].margins(x=.15, y=.15)
    axes[1].axhline(0,color='#777777',ls='--');axes[1].axvline(0,color='#777777',ls='--')
    axes[1].set(xlabel='Joint margin to 40 mrad (mrad)',ylabel='Upper height margin to 260 mm (mm)',title='Positive on both axes is required, not sufficient')
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('Local failures after calibrated PPO continuation — no interpolation or robustness claim')
    a.output.mkdir(exist_ok=False);fig.savefig(a.output/'local_failures.png',dpi=160);fig.savefig(a.output/'local_failures.pdf');plt.close(fig)
    (a.output/'diagnosis.json').write_text(json.dumps(dict(rows=rows,counts={k:sum(r['category']==k for r in rows) for k in styles},
        audit_sha256=run['audit_sha256'],checkpoint_sha256=run['checkpoint_sha256'],source_sha256=sha(__file__),
        scope='Only eight previously observed development cases. Margins are deterministic recorded values, not uncertainty estimates.'),indent=2))
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
