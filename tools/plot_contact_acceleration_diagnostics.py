"""Static research figure: recorded-state diagnostics, not closed-loop performance."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('outputs/contact_acceleration_diagnostic_v1');qp=Path('outputs/contact_acceleration_qp_soft_probe_v1')
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.4));colors=dict(original='#b34d3e',support='#247891')
    with np.load(root/'diagnostic.npz') as z:
        for label in ('original','support'):
            a=z[label];times=np.unique(a[:,0]);values=[a[a[:,0]==t,2].mean() for t in times]
            axes[0].plot(times,values,label=label+' reference',color=colors[label],linewidth=2)
    axes[0].axhline(0,color='#777777',linewidth=.8);axes[0].set(xlabel='Recorded simulation time (s)',ylabel='Desired normal acceleration (m/s²)',title='Right sole: PD target at recorded states')
    summary=json.loads((qp/'summary.json').read_text());audit=json.loads((qp/'independent_audit.json').read_text());assert audit['passed']
    with np.load(qp/'solutions.npz') as z:
        for label in ('original','support'):
            rows=[r for r in summary['rows'] if r['label']==label];times=[r['time_s'] for r in rows]
            before=[];after=[]
            for r in rows:
                key=f"{label}_{r['time_s']:.6g}"
                for suffix,values in [('desired_contact_acceleration',before),('solved_contact_acceleration',after)]:values.append(float(np.sqrt(np.mean(z[key+'_'+suffix]**2))))
            axes[1].plot(times,before,'o--',color=colors[label],alpha=.6,label=label+' PD')
            axes[1].plot(times,after,'s-',color=colors[label],label=label+' offline QP')
    axes[1].set(xlabel='Recorded simulation time (s)',ylabel='All contact acceleration RMS (m/s²)',title='Offline constrained acceleration/force allocation',yscale='log')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Contact consistency diagnostics — known +12.5 mm rod case',fontsize=13)
    fig.text(.5,.015,'Saved-state analysis only. QP is not a dynamic success result; no force/separation complementarity.',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.055,1,.94))
    for suffix in ('png','pdf'):fig.savefig(root/f'contact_consistency.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
