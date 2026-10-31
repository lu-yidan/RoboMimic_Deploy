"""Standalone scientific figure; requires numpy and matplotlib only."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);a=p.parse_args()
    s=json.loads((a.data/'summary.json').read_text());assert hashlib.sha256((a.data/'traces.npz').read_bytes()).hexdigest()==s['traces_sha256']
    with np.load(a.data/'traces.npz') as z:
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        for col,(field,threshold,unit) in enumerate([('closure_mrad',40.,'Maximum joint deviation (mrad)'),('rise_mm',260.,'Root rise (mm)')]):
            for row in range(2):
                ax=axes[row,col]
                for key,name,color in [('original','Selected parent','#b75242'),('modified','Search bias + waist/body targets','#24758b')]:
                    ts=z[key+'_time_s'];keep=ts>=(111.66 if row else 99.66)
                    ax.plot(ts[keep],z[key+'_'+field][keep],label=name,color=color,lw=1.7)
                ax.axhline(threshold,color='#555555',ls='--',lw=1,label='Original upper limit')
                ax.set_xlabel('Simulation time (s)');ax.set_ylabel(unit);ax.grid(alpha=.2)
                ax.set_title('Final 2 s acceptance window' if row else 'Complete closure segment')
                if not row:ax.axvspan(111.66,113.66,color='#ccddee',alpha=.25)
        axes[0,0].legend(fontsize=8)
        fig.suptitle('Known +15 mm case: full-cycle verification\n5/5 known cases; narrow margins; NOT a new PPO training gain',fontsize=12)
        fig.savefig(a.data/'closure_comparison.png',dpi=180);fig.savefig(a.data/'closure_comparison.pdf');plt.close(fig)

if __name__=='__main__':main()
