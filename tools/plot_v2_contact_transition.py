"""Inspect a recorded load transition at full physics cadence and 50 Hz."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--event-time',type=float,required=True);p.add_argument('--half-window',type=float,default=.2)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    audit=json.loads((a.run/'telemetry_audit.json').read_text());assert audit['passed']
    assert audit['telemetry_sha256']==sha(a.run/'physics_telemetry.npz')
    assert audit['trajectory_sha256']==sha(a.run/'trajectory.npz')
    with np.load(a.run/'physics_telemetry.npz') as file:z={k:file[k] for k in file.files}
    with np.load(a.run/'trajectory.npz') as file:x={k:file[k] for k in ('qacc_50hz','time_s')}
    t=z['time_s'];mask=abs(t-a.event_time)<=a.half_window
    samples=np.rint(x['time_s'][1:]/.0005).astype(int)-1
    selected=samples[abs(t[samples]-a.event_time)<=a.half_window]
    root=np.linalg.norm(z['solver_qacc'][:,:3],axis=1)
    force=np.linalg.norm(z['group_force_world_n'],axis=2)
    # Joint DOF 21 and motor 15 are the frozen 29-DoF G1 left shoulder pitch.
    shoulder=z['solver_qacc'][:,21];torque=z['motor_torque_nm'][:,15]
    peak_index=np.flatnonzero(mask)[np.argmax(root[mask])]
    row=dict(peak_root_acceleration_2000hz_m_s2=float(root[mask].max()),
        peak_root_acceleration_50hz_m_s2=float(root[selected].max()),
        peak_root_acceleration_time_s=float(t[peak_index]),
        peak_shoulder_acceleration_2000hz_rad_s2=float(abs(shoulder[mask]).max()),
        peak_shoulder_acceleration_50hz_rad_s2=float(abs(shoulder[selected]).max()),
        peak_hook_force_2000hz_n={g:float(force[mask,j].max()) for j,g in enumerate(audit['groups']) if 'hook' in g},
        peak_hook_force_50hz_n={g:float(force[selected,j].max()) for j,g in enumerate(audit['groups']) if 'hook' in g})
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(metrics=row,event_time_s=a.event_time,half_window_s=a.half_window,
        telemetry_audit_sha256=sha(a.run/'telemetry_audit.json'),source_sha256=sha(__file__),
        scope='Temporal association with the planned left-hook load transition, not a causal intervention. Raw 2000 Hz solver-stage data; 50 Hz points are exact subsamples at the saved policy times. Root acceleration is not COM acceleration; net hook force is not a single contact force. No new acceptance threshold.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (a.output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True,layout='constrained')
    ms=(t-a.event_time)*1000
    for ax,values,label in [(axes[0],root,'Root acceleration norm (m/s²)'),(axes[2],shoulder,'Left shoulder acceleration (rad/s²)'),(axes[3],torque,'Left shoulder motor torque (Nm)')]:
        ax.plot(ms[mask],values[mask],label='2000 Hz',color='tab:blue')
        ax.scatter(ms[selected],values[selected],label='50 Hz samples',color='tab:orange',s=17,zorder=3)
        ax.set_ylabel(label)
    for group,color in [('left_hook','tab:green'),('right_hook','tab:red')]:
        j=audit['groups'].index(group)
        axes[1].plot(ms[mask],force[mask,j],label=group,color=color)
        axes[1].scatter(ms[selected],force[selected,j],color=color,s=17,zorder=3)
    axes[1].set_ylabel('Net hook force norm (N)')
    for ax in axes:
        ax.axvline(0,color='gray',linestyle='--',alpha=.6);ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=8)
    axes[-1].set_xlabel('Time relative to planned load activation (ms)')
    fig.suptitle('T12, second cycle: left-hook loading\nExact replay of accepted v2 dynamics; short peaks remain between 50 Hz samples')
    fig.savefig(a.output/'load_transition.png',dpi=160);fig.savefig(a.output/'load_transition.pdf');plt.close(fig)
    print(json.dumps(row),flush=True)


if __name__=='__main__':main()
