"""Compare actual motor utilization and terminal tracking on the same reference."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args()
    fig,axs=plt.subplots(3,1,figsize=(11,9),layout='constrained')
    for run,label,color in [(a.baseline,'Previous compact controller','#74838c'),(a.candidate,'Torque objective + settling leg gain','#007d99')]:
        z=np.load(run/'trajectory.npz');ref=np.load(run/'reference.npz')
        # Native motor limits are stored by the frozen environment's contract.
        audit=json.loads((run/'independent_audit.json').read_text())
        peaks=audit['torque_peak_ratio_by_joint_50hz']
        # Position columns follow G1's floating base plus its 29 native joints.
        names=list(peaks);ankle=names.index('right_ankle_pitch_joint')
        t=z['time_s'];initial=ref['qpos'][0,7+ankle]
        axs[0].plot(t,np.rad2deg(z['qpos'][:,7+ankle]-initial),label=label,color=color,lw=1.1)
        mask=t>ref['time'][-1]-14
        axs[1].plot(t[mask],np.rad2deg(z['qpos'][mask,7+ankle]-initial),label=label,color=color,lw=1.3)
        selected=[n for n in names if 'wrist' in n or n=='right_shoulder_pitch_joint']
        x=np.arange(len(selected));offset=-.2 if run==a.baseline else .2
        axs[2].bar(x+offset,[100*peaks[n] for n in selected],width=.4,label=label,color=color)
        axs[2].set_xticks(x,[n.replace('_joint','').replace('_','\n') for n in selected],fontsize=8)
    axs[0].set_ylabel('Right ankle angle\nfrom initial (deg)');axs[0].set_xlabel('Simulation time (s)')
    axs[1].set_ylabel('Terminal ankle error (deg)');axs[1].set_xlabel('Simulation time (s)')
    axs[1].axhline(np.rad2deg(.04),ls='--',color='#bc4545',lw=1,label='Joint closure limit')
    axs[1].axhline(-np.rad2deg(.04),ls='--',color='#bc4545',lw=1)
    axs[2].set_ylabel('Peak motor limit use (%)');axs[2].axhline(100,ls='--',color='#bc4545',lw=1)
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Recorded single-cycle dynamics | unchanged reference, geometry and motor limits')
    for ext in ['png','pdf']:fig.savefig(a.candidate/('allocation_comparison.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
