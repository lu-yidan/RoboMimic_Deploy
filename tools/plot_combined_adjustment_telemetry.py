"""Plot verified force/acceleration block envelopes, preserving physical peaks."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);a=p.parse_args()
    s=json.loads((a.data/'summary.json').read_text());assert hashlib.sha256((a.data/'plot_data.npz').read_bytes()).hexdigest()==s['plot_data_sha256']
    with np.load(a.data/'plot_data.npz') as z:
        fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
        for column,start in enumerate([70.2,111.66]):
            for label,name,color in [('original',s['labels']['original'],'#b75242'),('modified',s['labels']['modified'],'#24758b')]:
                t=z[label+'_time_s'];mask=t>=start
                lo=z[label+'_root_acceleration_min'][mask];hi=z[label+'_root_acceleration_max'][mask]
                axes[0,column].fill_between(t[mask],lo,hi,color=color,alpha=.2)
                axes[0,column].plot(t[mask],z[label+'_root_acceleration_mean'][mask],color=color,lw=1,label=name)
                for row,indices in [(1,[0,1]),(2,[2,3])]:
                    for i in indices:
                        value=z[label+'_support_vertical_mean'][mask,i]
                        axes[row,column].plot(t[mask],value,color=color,ls='-' if i%2==0 else '--',lw=1,label=name+(' left' if i%2==0 else ' right'))
            axes[0,column].set_title('Landing and closure' if column==0 else 'Final 2 s')
            for row,title in enumerate(['Root acceleration norm (m/s2)','Foot vertical force (N)','Hook vertical force (N)']):
                axes[row,column].set_ylabel(title);axes[row,column].set_xlabel('Simulation time (s)');axes[row,column].grid(alpha=.2)
        axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=7,ncol=2)
        fig.suptitle('Known +15 mm: parent and joint/reference calibration\nMuJoCo 2000 Hz source; 100 Hz block envelopes for display',fontsize=12)
        fig.savefig(a.data/'physics_comparison.png',dpi=180);fig.savefig(a.data/'physics_comparison.pdf');plt.close(fig)

if __name__=='__main__':main()
