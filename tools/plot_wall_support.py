"""Plot the saved nominal PD comparison; uses the project's CAD plotting env."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(path):
    rows=list(csv.DictReader((path/'control_log.csv').open()))
    return {key:np.array([float(r[key]) for r in rows]) for key in
            ('simulation_time_s','root_error_m','tilt_rad','left_hook_vertical_n','right_hook_vertical_n',
             'left_foot_vertical_n','right_foot_vertical_n','total_contact_vertical_n','robot_weight_n')}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=ROOT/'outputs/wall_support_pd15_10s_v2')
    p.add_argument('--comparison',type=Path,default=ROOT/'outputs/wall_support_original_pd_10s_v2')
    a=p.parse_args();selected=load(a.run);baseline=load(a.comparison)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for data,label,color in [(baseline,'Original PD','#8b969e'),(selected,'Kp x1.5, Kd x sqrt(1.5)','#187c9f')]:
        t=data['simulation_time_s']
        axs[0,0].plot(t,data['root_error_m']*1000,label=label,color=color)
        axs[0,1].plot(t,np.rad2deg(data['tilt_rad']),label=label,color=color)
    axs[0,0].axhline(30,color='#b33b40',ls='--',label='30 mm criterion')
    axs[0,0].set(title='Pelvis displacement from initial pose',ylabel='mm');axs[0,0].legend(fontsize=8)
    axs[0,1].axhline(10,color='#b33b40',ls='--');axs[0,1].set(title='Pelvis tilt (criterion 10 deg)',ylabel='degrees')
    for key,label,color in [('left_hook_vertical_n','Left hook','#147b89'),('right_hook_vertical_n','Right hook','#53abac'),
                            ('left_foot_vertical_n','Left foot','#c97c24'),('right_foot_vertical_n','Right foot','#e5b666')]:
        axs[1,0].plot(selected['simulation_time_s'],selected[key],label=label,color=color)
    axs[1,0].set(title='Vertical contact force on robot (50 Hz samples)',ylabel='N');axs[1,0].legend(ncol=2,fontsize=8)
    axs[1,1].plot(selected['simulation_time_s'],selected['total_contact_vertical_n'],color='#187c9f',label='Contact force sum')
    axs[1,1].plot(selected['simulation_time_s'],selected['robot_weight_n'],color='#b33b40',ls='--',label='Model weight')
    axs[1,1].set(title='Total vertical support; final 2 s shaded',ylabel='N');axs[1,1].legend(fontsize=8)
    axs[1,1].axvspan(8,10,color='#d4e7e8',alpha=.4)
    for ax in axs.flat:ax.set(xlabel='Simulation time (s)',xlim=(0,10));ax.grid(alpha=.18)
    fig.suptitle('Rung-wall commissioning: fixed-target PD, not a trained climbing policy',fontweight='bold')
    fig.savefig(a.run/'support_metrics.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
