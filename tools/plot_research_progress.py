"""Evidence-backed comparison; retain failures and unchanged success counts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/research_progress_20261017'


def read(path):return json.loads((ROOT/path).read_text())


def main():
    OUT.mkdir(exist_ok=True)
    old=read('outputs/rung_error_grid_v2/results.json')
    new=read('outputs/clearance_back20_leg14_grid_v1/results.json')
    def code(r,legacy=False):
        if legacy and r['name']=='row01_x_-10mm':return 4
        if r['passed']:return 0
        if r['reason']!='complete':return 1
        if max(r['flight_contact_n'],r['other_contact_n'],r['self_contact_n'])>=1:return 2
        return 3
    colors=['#27866b','#ca4a4a','#df963b','#8061a8','#8b949e']
    names=['Strict pass','Allocation stopped','Completed with collision','Completed; endpoint failed','Initial overlap invalid']
    fig=plt.figure(figsize=(12,10),layout='constrained')
    gs=fig.add_gridspec(2,2,width_ratios=[1.3,1])
    ax=fig.add_subplot(gs[:,0])
    data=np.array([[code(a,True),code(b)] for a,b in zip(old,new)])
    ax.imshow(data,cmap=ListedColormap(colors),vmin=0,vmax=4,aspect='auto')
    ax.set_yticks(range(len(new)),[r['name'] for r in new],fontsize=9)
    ax.set_xticks([0,1],['Previous reference','Clearance + rearward stance'])
    ax.set_title('Same strict acceptance: 3/17 -> 3/17')
    ax.legend(handles=[Patch(color=c,label=n) for c,n in zip(colors,names)],loc='upper center',bbox_to_anchor=(.5,-.04),fontsize=8)
    ax2=fig.add_subplot(gs[0,1])
    case='row01_x_-5mm';a=next(r for r in old if r['name']==case);b=next(r for r in new if r['name']==case)
    x=np.arange(2);width=.35
    for shift,row,label,color in [(-width/2,a,'Previous','#a0a9b3'),(width/2,b,'New','#27866b')]:
        vals=[row['flight_contact_n'],row['other_contact_n']]
        bars=ax2.bar(x+shift,vals,width,label=label,color=color)
        ax2.bar_label(bars,fmt='%.2f',padding=3)
    ax2.set_xticks(x,['Swing contact','Other body contact']);ax2.set_ylabel('Peak force [N]')
    ax2.set_title('Foot rung X = -5 mm');ax2.legend();ax2.set_ylim(0,82)
    ax3=fig.add_subplot(gs[1,1]);x=np.arange(3)
    for shift,case,label,color in [(-.18,'row01_z_+5mm','Foot rung Z +5 mm','#27866b'),(.18,'row01_x_-10mm','Foot rung X -10 mm','#ca4a4a')]:
        vals=[read(f'outputs/{s}/{case}/summary.json')['hold_duration_s'] for s in ['unload_online_guard_v1','unload_lookahead100_v1','unload_lookahead250_v1']]
        bars=ax3.bar(x+shift,vals,.36,label=label,color=color);ax3.bar_label(bars,fmt='%.2f',padding=3)
    ax3.axhline(.5,color='black',ls='--',lw=1);ax3.set_ylim(0,.65)
    ax3.set_xticks(x,['On failure','100 ms lookahead','250 ms lookahead'],fontsize=8)
    ax3.set_ylabel('Actual hold duration [s]');ax3.set_title('Exact-model prediction; nominal never stopped')
    ax3.legend(fontsize=8)
    fig.suptitle('G1 climbing: clearance, bounded stopping, and remaining failures',fontsize=15)
    fig.text(.52,.01,'Caution: hand rung X -10 mm swing force increased 17.50 -> 20.28 N.\nSingle-foot PPO: zero/before/after all 5/5; no learning improvement shown.',fontsize=9)
    fig.savefig(OUT/'comparison.png',dpi=160);fig.savefig(OUT/'comparison.pdf');plt.close(fig)


if __name__=='__main__':main()
