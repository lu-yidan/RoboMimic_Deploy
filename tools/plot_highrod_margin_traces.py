import json
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=Path('outputs/highrod_height_mismatch_v1');data=json.loads((p/'margin_traces.json').read_text())
    fig,axes=plt.subplots(2,1,figsize=(9,6.7),sharex=True,layout='constrained')
    colors=['#31688e','#35a98a','#658b23','#d57b20','#b84343']
    for r,color in zip(data['rows'],colors):
        style='-' if r['accepted'] else '--'
        label=f"Actual +{r['true_height_mm']:g} mm ({'pass' if r['accepted'] else 'fail'})"
        axes[0].plot(r['time_s'],r['right_ankle_pitch_rad'],color=color,ls=style,label=label)
        axes[1].plot(r['time_s'],[1000*x for x in r['minimum_all_joint_margin_rad']],color=color,ls=style)
        if not r['accepted']:
            axes[0].scatter(r['time_s'][-1],r['right_ankle_pitch_rad'][-1],color=color,s=25,zorder=4)
            axes[1].scatter(r['time_s'][-1],1000*r['minimum_all_joint_margin_rad'][-1],color=color,s=25,zorder=4)
    axes[0].axhline(data['rows'][0]['lower_limit_rad'],color='black',lw=1,ls=':',label='Native ankle lower limit')
    axes[1].axhline(0,color='black',lw=1,ls=':')
    axes[1].axhline(20,color='#777777',lw=1,ls=':',label='Proposed penalty begins at 20 mrad')
    axes[0].set_ylabel('Right ankle pitch (rad)');axes[1].set_ylabel('Smallest native joint margin (mrad)')
    axes[1].set_xlabel('Simulation time (s)');axes[1].set_xlim(45,65)
    axes[0].legend(fontsize=8,loc='upper right',ncol=2);axes[1].legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.suptitle('Fixed +10 mm reference: higher actual rods reach ankle limit')
    fig.supxlabel('Zero actions, -3 deg normal bias. Saved states at 50 Hz; runtime limits checked at 2000 Hz.\nCurves stop at termination. Five discrete cases; no continuous safety guarantee.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(p/f'native_margin_traces.{ext}',dpi=170)
    plt.close(fig)
    sha=lambda file:hashlib.sha256(Path(file).read_bytes()).hexdigest()
    (p/'margin_plot_provenance.json').write_text(json.dumps(dict(data_sha256=sha(p/'margin_traces.json'),source_sha256=sha(__file__),png_sha256=sha(p/'native_margin_traces.png')),indent=2))
    print(str(p/'native_margin_traces.png'))


if __name__=='__main__':main()
