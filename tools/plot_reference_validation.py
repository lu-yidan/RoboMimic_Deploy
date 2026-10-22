"""Validation grid and endpoint closure, without treating samples as random trials."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    p=Path('outputs/support_reference_validation_v1');summary=json.loads((p/'validation_summary.json').read_text());rows=summary['rows']
    heights=sorted({r['offset_mm'][2] for r in rows});offsets=sorted({r['offset_mm'][0] for r in rows});grid=np.zeros((len(offsets),len(heights)))
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.4),gridspec_kw=dict(width_ratios=[1.05,1]))
    for yi,x in enumerate(offsets):
        closure=[]
        for xi,h in enumerate(heights):
            r=next(v for v in rows if v['offset_mm'][0]==x and v['offset_mm'][2]==h);grid[yi,xi]=int(r['passed']);c=r['endpoint']['joint_closure_rad']*1000;closure.append(c)
            axes[0].text(xi,yi,('PASS' if r['passed'] else 'FAIL')+f'\n{c:.1f} mrad',ha='center',va='center',fontsize=10,color='#222222')
        axes[1].plot(heights,closure,'o-',label=f'x offset {x:+g} mm',linewidth=1.7)
    axes[0].imshow(grid,cmap=ListedColormap(['#eac0a4','#b8d8cb']),vmin=0,vmax=1,aspect='auto')
    axes[0].set(xticks=range(len(heights)),xticklabels=[f'{h:g}' for h in heights],yticks=range(len(offsets)),yticklabels=[f'{x:+g}' for x in offsets],xlabel='Rod height offset (mm)',ylabel='Rod x offset (mm)',title='Strict acceptance: 9 / 12')
    axes[1].axhline(40.,color='#b34d3e',linestyle='--',label='Strict closure limit')
    axes[1].set(xlabel='Rod height offset (mm)',ylabel='Final 2 s maximum joint closure (mrad)',title='Joint closure is only one endpoint criterion',xticks=heights)
    axes[1].grid(alpha=.2);axes[1].legend(fontsize=8);axes[1].spines[['top','right']].set_visible(False)
    fig.suptitle('Frozen support-reference validation — full 113.66 s trajectories',fontsize=13)
    fig.text(.5,.012,'Deterministic interpolation/offset cases; oracle contact feedback. No statistical or hardware generalization claim.',ha='center',fontsize=8.5,color='#555555')
    fig.tight_layout(rect=(0,.05,1,.94))
    for suffix in ('png','pdf'):fig.savefig(p/f'validation_grid.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
