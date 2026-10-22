"""Source-derived figures; full-cycle and segment results kept separate."""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'outputs/research_progress_20261017'
    summaries=[json.loads((ROOT/'outputs'/p/'results.json').read_text()) for p in
        ['clearance_back20_leg14_grid_v1','tracking_relaxation_grid_v1','contact_frame_continued_grid_v1']]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    passed=[sum(r['passed'] for r in rows) for rows in summaries]
    axes[0].bar(['Fixed contacts','Tracking relaxation','Exact contact frames'],passed,color=['#8c99a7','#d2a044','#218c87'])
    axes[0].set_ylim(0,18);axes[0].set_yticks(range(0,18,2));axes[0].set_ylabel('Strict passes / 17 deterministic full cycles')
    for i,v in enumerate(passed):axes[0].text(i,v+.3,f'{v}/17',ha='center')
    axes[0].set_title('Same reference, scene grid and acceptance')
    results=json.loads((ROOT/'outputs/contact_bias_segments_v1/results.json').read_text());g=defaultdict(list)
    for r in results:g[r['trial'].split('__')[0]].append(r['success'])
    labels=list(g);values=[sum(g[k]) for k in labels]
    axes[1].barh(np.arange(len(labels)),values,color=['#b4514f' if v<3 else '#218c87' for v in values])
    axes[1].set_yticks(np.arange(len(labels)),[s.replace('_',' ').replace('minus','-').replace('plus','+') for s in labels])
    axes[1].invert_yaxis();axes[1].set_xlim(0,3.5);axes[1].set_xticks([0,1,2,3]);axes[1].set_xlabel('Successful load-transfer segments / 3')
    for i,v in enumerate(values):axes[1].text(v+.06,i,f'{v}/3',va='center')
    axes[1].set_title('Fixed calibration bias; 49–61.56 s only')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('G1 hook climbing: contact modeling and sensitivity',fontsize=15)
    fig.supxlabel('Exact contact identity and ideal force feedback remain assumptions. Counts are not population success probabilities.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'contact_research_v1.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    for tag,label,color in [('nominal','Nominal','#6b7280'),('minus5','Rod Z -5 mm','#2c8d91'),('minus10','Rod Z -10 mm','#b86c32')]:
        rows=json.loads((ROOT/f'outputs/contact_landing_{tag}_probe_v1/records.json').read_text())
        t=np.array([r['time_s'] for r in rows]);axes[0].plot(t,[1000*r['depth_m'] for r in rows],label=label,color=color)
        axes[1].plot(t,[r['lf_force_n'] for r in rows],color=color)
        axes[2].plot(t,[r['lf_normal_cap_n'] for r in rows],color=color)
    for ax in axes:ax.grid(alpha=.2);ax.axvline(79.72,color='k',ls=':',lw=.8);ax.axvline(81.72,color='k',ls=':',lw=.8)
    axes[0].set_ylabel('Foot search depth (mm)');axes[0].legend()
    axes[1].set_ylabel('Actual LF vertical force (N)');axes[1].axhline(10,color='gray',ls='--')
    axes[2].set_ylabel('QP per-point normal cap (N)');axes[2].set_xlabel('Simulation time (s)')
    fig.suptitle('Probe-loading diagnostic: 79–84 s recorded-prefix restarts')
    fig.supxlabel('The cap bounds allocated force, not measured force. These curves alone are not full-cycle validation.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'landing_segments_v1.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
