"""Plot all declared checkpoints, including the untrained initialization."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('screen',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=json.loads((args.screen/'summary.json').read_text());assert summary['completed']
    rows=summary['rows'];assert len(rows)==14 and rows[0]['label']=='initial' and rows[-1]['label']=='final'
    x=[0]+[int(row['label'].split('_')[1]) for row in rows[1:-1]]
    series=[('closure_rad',1000.,40.,'Worst joint closure (mrad)'),('rise_max_m',1000.,260.,'Maximum root rise (mm)')]
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    for axis,(key,scale,limit,label) in zip(axes,series):
        y=[row[key]*scale for row in rows[:-1]]
        axis.plot(x,y,'o-',color='#287a8b',ms=4,label='Saved before PPO update')
        axis.scatter([98304],[rows[-1][key]*scale],marker='X',s=90,color='#b35340',label='Final, after PPO update',zorder=4)
        axis.axhline(rows[0][key]*scale,color='#666666',ls=':',label='Initial search seed')
        axis.axhline(limit,color='#b35340',ls='--',lw=1,label='Acceptance limit')
        axis.set(xlabel='Additional collected transitions',ylabel=label)
        axis.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
        axis.ticklabel_format(axis='y',style='plain',useOffset=False);axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Known highest case: all checkpoints pass, initial margin remains best\nCached closure segment; development selection, not independent generalization',fontsize=11)
    args.output.mkdir(exist_ok=False)
    for suffix in ('png','pdf'):fig.savefig(args.output/f'checkpoint_screen.{suffix}',dpi=180)
    metadata=dict(summary_sha256=hashlib.sha256((args.screen/'summary.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),all_candidates_shown=True)
    (args.output/'plot_manifest.json').write_text(json.dumps(metadata,indent=2))
    (args.output/'plot_source.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
