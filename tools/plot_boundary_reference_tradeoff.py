"""Plot recorded reference-probe tradeoffs, without inferring global feasibility."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--search',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 data=json.loads((a.search/'progress.json').read_text());rows=data['rows']
 for r in rows:assert sha(r['file'])==r['trajectory_sha256']
 a.output.mkdir(exist_ok=False)
 styles={'base':('Parent','black','*',85),'axis3_minus':('Root command -0.1','#2166ac','v',48),'axis3_plus':('Root command +0.1','#b2182b','^',48),'common_0.25':('Coordinated candidate','#762a83','D',36)}
 fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
 for ax,case in zip(axs.flat,['boundary_04','boundary_05','boundary_06','boundary_07']):
  ax.add_patch(Rectangle((37.2,259.91),2.8,.09,facecolor='#d9f0d3',alpha=.65,zorder=0))
  ax.axvline(40,color='#777777',lw=1,ls='--');ax.axhline(260,color='#777777',lw=1,ls='--')
  group=[r for r in rows if r['case']==case]
  for r in group:
   label,color,marker,size=styles.get(r['label'],('Other reference probes','#999999','x',22))
   ax.scatter(r['closure_mrad'],r['rise_max_mm'],c=color,marker=marker,s=size,label=label,zorder=4 if r['label']=='base' else 3)
  parent=next(r for r in group if r['label']=='base')
  for label in ['axis3_minus','axis3_plus']:
   r=next(r for r in group if r['label']==label)
   ax.plot([parent['closure_mrad'],r['closure_mrad']],[parent['rise_max_mm'],r['rise_max_mm']],color=styles[label][1],lw=1)
  ax.set_title(case+' (parent '+('pass' if parent['success'] else 'fail')+')',fontsize=11)
  ax.set_xlim(37.2,41.6);ax.set_ylim(259.91,260.18);ax.grid(alpha=.2)
 for ax in axs[-1]:ax.set_xlabel('Maximum joint closure error (mrad)')
 for ax in axs[:,0]:ax.set_ylabel('Maximum final root rise (mm)')
 handles,labels=axs[0,0].get_legend_handles_labels();unique=dict(zip(labels,handles))
 fig.legend(unique.values(),unique.keys(),loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(.5,.055))
 fig.suptitle('Reference-only changes trade height margin against ankle closure',fontsize=14)
 fig.text(.5,.015,'Observed full cycles only. Shading shows two bounds, not all task conditions. Known development cases.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.15,1,.95))
 fig.savefig(a.output/'reference_tradeoff.png',dpi=180);fig.savefig(a.output/'reference_tradeoff.pdf')
 (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
 (a.output/'summary.json').write_text(json.dumps(dict(progress_sha256=sha(a.search/'progress.json'),source_sha256=sha(__file__),rows=rows,scope='Observed saved results; no new simulation or global infeasibility claim.'),indent=2))
if __name__=='__main__':main()

