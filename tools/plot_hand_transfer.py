"""Plot measured simulator trajectories; run with the CAD/plotting environment."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    with (args.run/'control_log.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    column=lambda key:np.array([float(r[key]) for r in rows])
    t=column('time_s');fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True,layout='constrained')
    axs[0].plot(t,1000*(column('right_hook_height_m')-1.5432),label='Right hook vertical displacement')
    axs[0].plot(t,1000*(column('right_hook_x_m')-.385607427),label='Right hook X displacement')
    axs[0].set_ylabel('Displacement (mm)');axs[0].legend(loc='upper left')
    for name,label in [('right_hook_vertical_n','Right hook'),('left_hook_vertical_n','Left hook'),('left_foot_vertical_n','Left foot'),('right_foot_vertical_n','Right foot')]:
        axs[1].plot(t,column(name),label=label)
    axs[1].axhline(10,color='grey',ls=':',label='Loaded threshold: 10 N')
    axs[1].set_ylabel('Vertical contact force (N)');axs[1].legend(ncol=3)
    axs[2].plot(t,1000*column('root_error_m'),label='Root displacement')
    axs[2].plot(t,1000*column('ik_position_error_m'),label='IK residual')
    axs[2].set_ylabel('Distance (mm)');axs[2].set_xlabel('Simulation time (s)');axs[2].legend()
    for ax in axs:
        ax.grid(alpha=.25)
        for x in [2,4,6,10,12,14]:ax.axvline(x,color='grey',alpha=.2)
        ax.axvspan(14,18,color='#51a69a',alpha=.10)
    fig.suptitle(args.run.name+' | scripted controller, measured free-base dynamics')
    fig.savefig(args.run/'motion_metrics.png',dpi=160)


if __name__=='__main__':main()
