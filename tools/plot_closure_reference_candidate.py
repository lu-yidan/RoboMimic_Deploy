"""Plot complete candidate comparisons with their changed reference authority explicit."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.compare_audited_closure_candidates import load_run,sha


def main():
 p=argparse.ArgumentParser()
 for name in ('before','after','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();before=load_run(a.before);after=load_run(a.after)
 assert before['cases']==after['cases'] and before['reference_sha256']==after['reference_sha256']
 assert before['checkpoint_sha256']==after['checkpoint_sha256']
 assert before['adaptive_reference_command'] is None and len(after['adaptive_reference_command'])==5
 names=list(before['cases']);labels=[name.replace('foot_z_plus','z').replace('x-0.1_z15','x-0.1 / z15').replace('x+0.1_z15','x+0.1 / z15') for name in names]
 x=np.arange(len(names));fig,axes=plt.subplots(2,1,figsize=(10,6.8),layout='constrained',sharex=True)
 for run,color,label,offset in [(before,'#287a8b','Fixed waist/body seed',-.12),(after,'#b35340','Added constant fore-aft/waist reference',.12)]:
  axes[0].plot(x+offset,[40-run['rows'][k]['closure_mrad'] for k in names],'o-',color=color,label=label)
  axes[1].plot(x+offset,[260-run['rows'][k]['rise_max_mm'] for k in names],'o-',color=color,label=label)
 for axis in axes:axis.axhline(0,color='#555',ls='--',lw=1);axis.grid(alpha=.2)
 axes[0].set_ylabel('Joint closure margin (mrad)');axes[1].set_ylabel('Upper root-rise margin (mm)')
 axes[1].set_xticks(x,labels,rotation=18,ha='right');axes[1].set_xlabel('Known geometry offsets in mm; development cases')
 axes[0].legend(fontsize=8)
 counts=[sum(r['success'] for r in run['rows'].values()) for run in (before,after)]
 fig.suptitle(f'Fixed reference search: {counts[0]}/{len(names)} to {counts[1]}/{len(names)} complete cycles\nSame actor checkpoint; additional reference authority, no new PPO learning',fontsize=12)
 a.output.mkdir(exist_ok=False)
 for suffix in ('png','pdf'):fig.savefig(a.output/f'candidate_comparison.{suffix}',dpi=180)
 (a.output/'manifest.json').write_text(json.dumps(dict(before=before,after=after,source_sha256=sha(__file__),scope='All eight audited full cycles. These plots show two endpoint constraints; full success also requires the other unchanged task checks.'),indent=2))
 (a.output/'plot_source.py').write_bytes(Path(__file__).read_bytes())

if __name__=='__main__':main()
