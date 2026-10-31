"""Display signed collision distances at fixed old poses, never success labels."""
import argparse
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.preflight.read_text());assert data['passed'] and data['cases']==16
    rows=data['rows'];xs=sorted({r['offsets']['rung_06'][0]*1000 for r in rows})
    zs=sorted({r['offsets']['rung_06'][2]*1000 for r in rows})
    matrices={};ranges={}
    for group in ('right_hook','left_hook'):
        values=np.full((4,4),np.nan)
        for row in rows:
            x,_,z=row['offsets']['rung_06'];values[zs.index(z*1000),xs.index(x*1000)]=row['fixed_baseline_pose_gaps'][group]['distance_mm']
        assert np.isfinite(values).all();matrices[group]=values
        ranges[group]=[float(values.min()),float(values.max())]
    vmin=min(v.min() for v in matrices.values());vmax=max(v.max() for v in matrices.values())
    norm=TwoSlopeNorm(vmin=min(vmin,-.001),vcenter=0.,vmax=vmax)
    fig,axes=plt.subplots(1,2,figsize=(11,5.5),layout='constrained')
    for ax,(group,values),when in zip(axes,matrices.items(),(127.66,140.66)):
        mesh=ax.imshow(values,origin='lower',cmap='coolwarm_r',norm=norm)
        for iz in range(4):
            for ix in range(4):
                value=values[iz,ix]
                ax.text(ix,iz,f'{value:.3f}',ha='center',va='center',fontsize=11,
                    color='white' if value>vmax*.7 else 'black')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=[f'{x:.3f}' for x in xs],
            yticklabels=[f'{z:.3f}' for z in zs],xlabel='rung_06 X offset (mm)',ylabel='rung_06 Z offset (mm)',
            title=f'{group.replace("_"," ").title()} at old T12 pose, t = {when:.2f} s')
    fig.colorbar(mesh,ax=axes,label='Signed hook–rod collision distance (mm)',shrink=.85)
    fig.suptitle('Static compiled-scene preflight: fixed old pose, 16 geometries\nPositive = gap; negative = overlap. No new dynamics or success evidence.',fontsize=12)
    a.output.mkdir(parents=True,exist_ok=False)
    fig.savefig(a.output/'signed_gaps.png',dpi=160);fig.savefig(a.output/'signed_gaps.pdf');plt.close(fig)
    report=dict(preflight_sha256=hashlib.sha256(a.preflight.read_bytes()).hexdigest(),ranges_mm=ranges,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physics_steps=0,
        scope='Saved nominal poses are held fixed while the scene changes. A negative distance indicates overlap at that reused pose, not observed penetration during a new rollout. No geometry was rejected or retuned from this plot.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (a.output/'plot_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(report))


if __name__=='__main__':main()
