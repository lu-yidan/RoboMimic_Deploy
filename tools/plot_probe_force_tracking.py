"""Show the first later-cycle collision before the feedback confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=json.loads(a.analysis.read_text());run=Path(d['run'])
    assert sha(run/'physics_telemetry.npz')==d['telemetry_sha256']
    record=json.loads((run/'hook_admission_record.json').read_text())
    with np.load(run/'physics_telemetry.npz') as z:
        times=z['time_s'];forces=z['group_force_world_n']
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    details={}
    for ax,g in zip(axes,('right_hook','left_hook')):
        rows=[r for r in d['rows'] if r['cycle']==2 and r['group']==g]
        onset=rows[0]['activation_s'];confirmed=next(r['confirmed_s'] for r in rows if r['confirmed_s'] is not None)
        mask=(times>=onset-.02)&(times<=onset+.16)
        x=times[mask]-onset;y=forces[mask,record['groups'].index(g),2]
        ax.plot(x*1000,y,label='Actual net vertical force (2 kHz)',color='#bb4141',lw=1.3)
        limited=[r for r in rows if r['time_s']<=onset+.165]
        tx=np.array([r['time_s']-onset for r in limited])*1000
        ax.step(tx,[r['modeled_total_vertical_cap_n'] for r in limited],where='post',
            label='Modeled total vertical cap',color='#386cb0',ls='--')
        ax.step(tx,[r['modeled_vertical_force_n'] for r in limited],where='post',
            label='QP modeled vertical force',color='#418064',lw=1)
        ax.axvline((confirmed-onset)*1000,color='#555555',ls=':',label='Feedback confirmation')
        j=int(np.argmax(y));ax.annotate(f'{y[j]:.1f} N',xy=(x[j]*1000,y[j]),xytext=(12,-5),
            textcoords='offset points',fontsize=10)
        ax.set(xlim=(-20,160),xlabel='Time since planned activation (ms)',ylabel='World-vertical force (N)',
            title=g.replace('_',' ').title()+' — cycle 2')
        ax.grid(alpha=.2)
        details[g]=dict(activation_s=onset,confirmation_delay_ms=(confirmed-onset)*1000,
                       plotted_peak_vertical_n=float(y[j]),plotted_peak_time_s=float(times[mask][j]))
    axes[0].legend(fontsize=8,loc='upper right')
    fig.suptitle('T12 candidate: modeled loading cap does not bound initial collision force\n'
                 'Existing recorded run; ideal feedback; no new controller or hardware safety claim',fontsize=12)
    a.output.mkdir(parents=True,exist_ok=False)
    fig.savefig(a.output/'initial_collision.png',dpi=160);fig.savefig(a.output/'initial_collision.pdf');plt.close(fig)
    (a.output/'plot_source.py').write_bytes(Path(__file__).read_bytes())
    (a.output/'manifest.json').write_text(json.dumps(dict(analysis_sha256=sha(a.analysis),
        source_sha256=sha(Path(__file__)),details=details,telemetry_sha256=d['telemetry_sha256']),indent=2),encoding='utf8')
    print(json.dumps(details))


if __name__=='__main__':main()
