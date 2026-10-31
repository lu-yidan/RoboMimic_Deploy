"""Plot every declared outcome and explicitly selected complete-case contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ARMS=('adopted_v2','contact_confirmed','timed_open_loop')
LABELS=('Adopted v2','Contact confirmation','Fixed-delay loading')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(source,output):
    data=json.loads(source.read_text(encoding='utf8'))
    cases=data['cases']
    assert data['declared_cases']==len(cases)==16 and data['declared_records']==48
    assert len({c['case'] for c in cases})==16
    xs=sorted({c['offsets']['rung_06'][0]*1000 for c in cases})
    zs=sorted({c['offsets']['rung_06'][2]*1000 for c in cases})
    assert len(xs)==len(zs)==4
    output.mkdir(parents=True,exist_ok=False)
    colors=ListedColormap(['#a5aab3','#c05252','#438a72'])
    norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    fig,axes=plt.subplots(1,3,figsize=(14,5.5),layout='constrained')
    counts={}
    for ax,arm,label in zip(axes,ARMS,LABELS):
        values=np.full((4,4),np.nan);annotations={}
        for c in cases:
            r=c['arms'][arm]
            x,_,z=c['offsets']['rung_06'];ix=xs.index(x*1000);iz=zs.index(z*1000)
            integrity=bool(r.get('integrity_passed'))
            code=2 if r['passed'] else 1 if integrity else 0
            values[iz,ix]=code
            annotations[(ix,iz)]=('PASS' if code==2 else 'FAIL' if code==1 else 'NO PASS')
        assert np.isfinite(values).all()
        counts[arm]=sum(c['arms'][arm]['passed'] for c in cases)
        assert counts[arm]==data['passed_counts'][arm]
        ax.imshow(values,origin='lower',cmap=colors,norm=norm)
        for (ix,iz),label_cell in annotations.items():
            ax.text(ix,iz,label_cell,ha='center',va='center',color='white',fontsize=10,fontweight='bold')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=[f'{x:.3f}' for x in xs],
               yticklabels=[f'{z:.3f}' for z in zs],xlabel='rung_06 X offset (mm)',
               ylabel='rung_06 Z offset (mm)',title=f'{label}: {counts[arm]}/16 pass')
    fig.suptitle('All predeclared cases, same policy and reference, no retraining\n'
                 'Red = audited task failure; gray = no successful audit or first-cycle gate failure',fontsize=12)
    fig.savefig(output/'all_case_outcomes.png',dpi=160)
    fig.savefig(output/'all_case_outcomes.pdf');plt.close(fig)
    selected=[c for c in cases if c['all_three_complete_and_audited']]
    excluded=[c['case'] for c in cases if not c['all_three_complete_and_audited']]
    # Complete-case metrics include audited complete task failures. Truncated
    # traces are never assigned zero or treated as peak reductions.
    metrics=[('root_acceleration_peak_m_s2','Root acceleration peak (m/s²)'),
             ('joint_acceleration_peak_rad_s2','Joint acceleration peak (rad/s²)'),
             ('left_hook','Left hook force peak (N)'),('right_hook','Right hook force peak (N)'),
             ('closure_mrad','Terminal closure error (mrad)'),
             ('summed_squared_torque_nm2_s','Squared torque integral (N²m²s)')]
    deltas={}
    if selected:
        fig,axes=plt.subplots(3,2,figsize=(13,11),layout='constrained')
        case_labels=[c['case'].replace('hook_grid_','') for c in selected]
        for ax,(key,title) in zip(axes.flat,metrics):
            def value(c,arm):
                m=c['arms'][arm]['complete_cycle_metrics']['metrics']
                return m['support_net_force_peaks_n'][key] if key in ('left_hook','right_hook') else m[key]
            baseline=np.array([value(c,ARMS[0]) for c in selected])
            deltas[key]={}
            for arm,label,color,shift in zip(ARMS[1:],LABELS[1:],('#386cb0','#d48828'),(-.12,.12)):
                y=np.array([value(c,arm) for c in selected])-baseline
                assert np.isfinite(y).all()
                ax.scatter(np.arange(len(y))+shift,y,label=label,color=color,s=28)
                deltas[key][arm]={c['case']:float(v) for c,v in zip(selected,y)}
            ax.axhline(0,color='#454545',lw=.8);ax.grid(axis='y',alpha=.25)
            ax.set(title=title,ylabel='Candidate minus adopted v2',xticks=range(len(selected)),
                   xticklabels=case_labels)
            ax.tick_params(axis='x',rotation=65,labelsize=8)
        axes.flat[0].legend(fontsize=9)
        fig.suptitle(f'Complete second cycle only: matched records for {len(selected)}/16 cases\n'
                     'Negative = lower metric, not necessarily safer. Excluded cases remain in outcome counts.\n'
                     'Squared torque is an effort proxy, not energy; deterministic development grid.',fontsize=12)
        fig.savefig(output/'complete_case_deltas.png',dpi=160)
        fig.savefig(output/'complete_case_deltas.pdf');plt.close(fig)
    report=dict(summary_sha256=sha(source),source_sha256=sha(Path(__file__)),
                passed_counts=counts,metric_case_count=len(selected),excluded_from_metric_comparison=excluded,
                deltas=deltas,metric_scope='Complete audited traces in all three arms only; this selection can bias metric comparisons. Full 16-case outcome denominator is separate. No confidence intervals or random-wall generalization.',
                artifacts={p.name:sha(p) for p in output.iterdir() if p.suffix in ('.png','.pdf')})
    (output/'plot_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (output/'plot_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(passed_counts=counts,metric_cases=len(selected),output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plot(a.summary.resolve(),a.output.resolve())
