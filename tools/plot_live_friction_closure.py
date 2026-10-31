"""Plot audited actual velocity response and terminal rise, without smoothing states."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    fig,axes=plt.subplots(2,1,figsize=(9,7),layout='constrained')
    for name,label in [('live_closure_trace_20261023_v1','Original friction model'),('measured_friction_closure_20261023_v1','Previous-command friction estimate')]:
        p=Path('outputs')/name/'gain0_drop1_foot_z_plus15'
        with np.load(p/'live_trace.npz') as z:
            t=z['post_step_time'][1:];a=np.diff(z['post_step_qvel'][:,2])/np.diff(z['post_step_time'])
            n=len(t)//100*100
            axes[0].plot(t[:n].reshape(-1,100).mean(axis=1),a[:n].reshape(-1,100).mean(axis=1),label=label)
        with np.load(p/'trajectory.npz') as z:
            mask=z['time_s']>=99.66
            axes[1].plot(z['time_s'][mask],1000*(z['qpos'][mask,2]-z['qpos'][0,2]),label=label)
    axes[0].set(ylabel='Integrated root acceleration (m/s²)',title='Known +15 mm case: continuous physical response (50 ms means)')
    axes[1].axhline(260,color='black',linestyle='--',label='Original upper acceptance limit')
    axes[1].set(xlabel='Simulation time (s)',ylabel='Root rise (mm)')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    p=Path('outputs/live_friction_comparison_20261023_v1')
    fig.savefig(p/'continuous_response.png',dpi=160);fig.savefig(p/'continuous_response.pdf')
    (p/'plot_source.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
