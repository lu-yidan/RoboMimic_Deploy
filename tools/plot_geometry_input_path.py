"""Display exact geometry projection weights without claiming behavior from nonzero weights."""
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
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=json.loads(a.audit.read_text());fig,axes=plt.subplots(1,2,figsize=(12,4.9),layout='constrained',sharey=True)
    labels=[f'{g}\n{axis}' for g in ('foot 1','foot 2','hand 1','hand 2') for axis in 'XYZ']
    for ax,(key,label) in zip(axes,(('prefix','Frozen prefix actor'),('suffix','Frozen suffix actor'))):
        values=np.asarray(d['actors'][key]['projection_max_absolute_by_input'])
        assert len(values)==12 and not values[6:].any()
        ax.axvspan(5.5,11.5,color='#f2e2e2',alpha=.6)
        ax.bar(range(12),values,color=['#386cb0']*6+['#b44b4b']*6,width=.65)
        for i,v in enumerate(values):ax.text(i,v+.00055,f'{v:.4f}' if v else '0',ha='center',fontsize=8)
        for x in (2.5,5.5,8.5):ax.axvline(x,color='#bbbbbb',lw=.6)
        ax.set(xticks=range(12),xticklabels=labels,title=label,ylim=(0,.025),
            ylabel='Maximum absolute projection weight per input')
        ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Hand-rod offset channels are exactly zero in both frozen checkpoints\n'
                 'Other 136 state/reference/contact features remain. Nonzero weight alone does not prove useful behavior.',fontsize=12)
    a.output.mkdir(parents=True,exist_ok=False)
    fig.savefig(a.output/'input_projection.png',dpi=160);fig.savefig(a.output/'input_projection.pdf');plt.close(fig)
    (a.output/'plot_source.py').write_bytes(Path(__file__).read_bytes())
    (a.output/'manifest.json').write_text(json.dumps(dict(audit_sha256=sha(a.audit),
        source_sha256=sha(Path(__file__)),figure_sha256=sha(a.output/'input_projection.png')),indent=2),encoding='utf8')

if __name__=='__main__':main()
