"""Standalone scientific figure of discrete, audited endpoint-gain experiments."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'outputs/highrod_gain_sensitivity_v1/paired_summary.json';data=json.loads(source.read_text());assert data['integrity_passed']
rows=data['results'];gains=[r['gain'] for r in rows]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(11,4.4),gridspec_kw={'width_ratios':[1,1.15]})
ax=axes[0];values=[r['endpoint_peak_rad'] for r in rows]
ax.bar(range(4),values,color=['#24978d' if r['task_passed'] else '#c95a59' for r in rows],width=.6)
ax.axhline(.04,color='#333333',ls='--',lw=1,label='Original acceptance limit')
for i,(r,y) in enumerate(zip(rows,values)):ax.text(i,y+.002,f"{y:.4f}\n{'PASS' if r['task_passed'] else 'FAIL'}",ha='center',fontsize=9)
ax.set(xticks=range(4),xticklabels=[str(g) for g in gains],xlabel='Endpoint correction gain',ylabel='Largest absolute joint error (rad)',ylim=(0,.09),title='A. Closure during final 2 s')
ax.legend(loc='upper right',frameon=False,fontsize=8)
ax=axes[1]
for joint,label,color in [('left_ankle_pitch_joint','Left ankle pitch','#3674aa'),('right_wrist_pitch_joint','Right wrist pitch','#d17d31'),('left_knee_joint','Left knee','#777777')]:
    values=[next(abs(j['peak_error_rad']) for j in r['joints'] if j['joint']==joint) for r in rows]
    ax.plot(gains,values,'o-',label=label,color=color,lw=1.5)
ax.axhline(.04,color='#333333',ls='--',lw=1)
ax.set(xlabel='Endpoint correction gain',ylabel='Absolute joint error (rad)',title='B. Error shifts across joints',xticks=gains,ylim=(0,.085));ax.legend(frameon=False,fontsize=9)
fig.suptitle('Known +10 mm foot rod: endpoint correction sensitivity',fontsize=13,y=.99)
fig.text(.5,.015,'Four discrete tests; lines are visual guides. Same scene, control and physical prefix until 99.66 s. No interval robustness claim.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.065,1,.95])
out=ROOT/'outputs/research_progress_20261017/highrod_gain_sensitivity_v1'
for suffix in ['.png','.pdf']:fig.savefig(out.with_suffix(suffix),dpi=180)
out.with_suffix('.json').write_text(json.dumps(dict(data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
print(out.with_suffix('.png'))
