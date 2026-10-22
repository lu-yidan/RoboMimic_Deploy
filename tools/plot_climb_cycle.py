"""Scientific summary of the recorded cycle, including failed runs."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with (a.run/'control_log.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    summary=json.loads((a.run/'summary.json').read_text())
    def col(k):return np.array([float(r[k]) for r in rows])
    t=col('time_s');fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True,layout='constrained')
    ref=np.load(a.run/'reference.npz');target_cm=100*(ref['qpos'][-1,2]-ref['qpos'][0,2])
    axes[0].plot(t,col('base_rise_m')*100,label='Recorded rise');axes[0].axhline(target_cm,ls='--',color='black',lw=1,label='Target rise')
    axes[0].set_ylabel('Base rise (cm)');axes[0].legend(loc='upper left')
    axes[1].plot(t,col('base_error_m')*1000);axes[1].axhline(30,ls='--',color='red',lw=1)
    axes[1].set_ylabel('Base tracking (mm)')
    for key,label in [('left_hook','Left hook'),('right_hook','Right hook'),('left_foot','Left foot'),('right_foot','Right foot')]:
        axes[2].plot(t,col(key+'_vertical_n'),label=label,lw=1)
    axes[2].set_ylabel('Vertical contact (N)');axes[2].legend(ncol=4,loc='upper left')
    axes[3].plot(t,col('max_joint_error_rad'));axes[3].set_ylabel('Max joint error (rad)');axes[3].set_xlabel('Simulation time (s)')
    events=json.loads((a.run/'phase_events.json').read_text())
    for ax in axes:
        ax.grid(alpha=.2)
        for event in events:
            if event['phase'] in ['rh_lift','lh_lift','step_preparation','rf_unload','lf_unload','body_rise','rf_straighten_unload','lf_straighten_unload','final_center','final_hold']:
                ax.axvline(event['time_s'],color='grey',alpha=.2,lw=.8)
    fig.suptitle('G1 torque-driven rung cycle | '+('PASS' if summary.get('passed') else 'NOT PASSED')+' | recorded simulation, not a trained policy')
    for suffix in ['png','pdf']:fig.savefig(a.run/('cycle_metrics.'+suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
