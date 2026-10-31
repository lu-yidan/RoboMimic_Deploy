"""Plot offline clearance diagnosis and preserve the real failure endpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.compare_audited_closure_candidates import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    s=json.loads((a.directory/'summary.json').read_text());assert sha(a.directory/'clearance.npz')==s['data_sha256']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    with np.load(a.directory/'clearance.npz') as data:
        for label,name,color in [('successful','Known +15 mm','#24758b'),('failure','Fresh B03 (failed)','#b75242')]:
            for ax in axes:
                ax.plot(data[label+'_reference_time_s'],1000*data[label+'_reference_distance_m'],'--',color=color,alpha=.65,label=name+' scripted')
                ax.plot(data[label+'_recorded_time_s'],1000*data[label+'_recorded_distance_m'],color=color,label=name+' recorded')
        end=data['failure_recorded_time_s'][-1]
        for ax in axes:
            ax.axhline(0,color='#333333',lw=1)
            ax.axvline(end,color='#b75242',lw=1,ls=':')
            ax.set(xlabel='Simulation time (s)',ylabel='Sole-to-target-rod distance (mm)')
            ax.grid(alpha=.2)
        axes[0].set_title('Left-foot swing interval')
        axes[1].set(xlim=(76.8,78.1),ylim=(-3,22),title='Recorded failure at 77.541 s')
        axes[0].legend(fontsize=8)
    fig.suptitle('Offline geometry of scripted poses and recorded physics\nScripted pose excludes actor/controller additions; failed recording is not extrapolated.',fontsize=11)
    fig.savefig(a.directory/'clearance.png',dpi=180);fig.savefig(a.directory/'clearance.pdf');plt.close(fig)
    (a.directory/'plot_source.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
