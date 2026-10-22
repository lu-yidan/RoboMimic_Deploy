"""Plot measured support forces and logged memory bounds for the full-cycle pair."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    suite=ROOT/'outputs/window_memory_fullcycles_v1';out=ROOT/'outputs/research_progress_20261017'
    audit=json.loads((suite/'paired_audit.json').read_text());assert audit['passed']
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
    sources={}
    for mode,label,color in [('raw','Raw biased normal','#c44e52'),('memory','Bounded contact memory','#2a9d8f')]:
        run=suite/(mode+'__row01_z_+5mm');path=run/'control_log.csv'
        rows=list(csv.DictReader(path.open()));t=np.array([float(r['time_s']) for r in rows])
        mask=(t>=53.2)&(t<=54.2)
        for ax,key in zip(axes[:2],['left_hook_vertical_n','left_foot_vertical_n']):
            ax.plot(t[mask],[float(r[key]) for r,keep in zip(rows,mask) if keep],label=label,color=color,linewidth=1.6,linestyle='--' if mode=='raw' else '-',zorder=3 if mode=='raw' else 2)
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    events=json.loads((suite/'memory__row01_z_+5mm/memory_events.json').read_text())
    times=np.array([e['time_s'] for e in events]);gaps=np.array([e['gap_m'] for e in events])*1e6
    axes[2].scatter(times,gaps,s=55,color='#2a9d8f',zorder=4,label='Cached allocation point used')
    axes[2].axhline(100,color='gray',linestyle=':',label='Gap gate:100 micrometres')
    for ax in axes:
        ax.axvline(53.7,color='#c44e52',linestyle='--',alpha=.7)
        ax.grid(alpha=.2)
    axes[0].axhline(10,color='gray',linestyle=':',alpha=.7)
    axes[0].annotate('Raw controller stops',xy=(53.7,32),xytext=(53.81,49),arrowprops=dict(arrowstyle='->',color='#c44e52'),color='#c44e52',fontsize=10)
    axes[0].set_ylabel('Left hook vertical force (N)')
    axes[1].set_ylabel('Left foot vertical force (N)')
    axes[2].set_ylabel('Recorded gap (micrometres)')
    axes[2].set_xlabel('Continuous simulation time (s)')
    axes[0].legend(loc='upper right',fontsize=9)
    axes[2].legend(loc='upper right',fontsize=9)
    axes[2].set_ylim(-3,110);axes[2].set_xlim(53.2,54.2)
    fig.suptitle('Continuous-cycle recovery: foot rod Z+5 mm, normal bias -3 deg\nForces sampled at 50 Hz; 4 memory uses, max 10 ms; exact geometric-gap gate')
    for extension in ['png','pdf']:fig.savefig(out/('memory_window_recovery_v1.'+extension),dpi=160)
    (out/'memory_window_recovery_v1.json').write_text(json.dumps(dict(sources=sources,scope='Measured contact forces, not planned QP forces. Raw run stops53.7s; memory run completes110.72s. Bias only49..61.56s. Gap event logging does not independently verify high-rate historical force gate.'),indent=2))


if __name__=='__main__':main()
