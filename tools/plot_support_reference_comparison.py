"""Visual summary of audited reference repair and the remaining endpoint failure."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('outputs/support_retarget_calibration_v1');selection=json.loads((root/'selection.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained');sources={}
    for directory,label,color,offset in [('highrod_height_mismatch_v1','Original reference','#8a8c92',-.3),('support_retarget_calibration_v1','Support retarget','#216a9c',.3)]:
        heights=[];margins=[];closures=[]
        for h in (5.,7.5,10.,12.5,15.):
            p=Path('outputs')/directory/f'true_height_{h:g}mm';s=json.loads((p/'summary.json').read_text());a=json.loads((p/'independent_audit.json').read_text())
            assert a['integrity_passed'] and a['summary_sha256']==sha(p/'summary.json')
            sources[str(p/'independent_audit.json')]=sha(p/'independent_audit.json')
            axes[0].barh(h+offset,s['time_s'],height=.5,color=color,label=label if h==5 else None,alpha=.9)
            axes[0].scatter(s['time_s'],h+offset,marker='o' if a['independently_accepted'] else 'x',color='#1f6536' if a['independently_accepted'] else '#bc433c',s=35,zorder=4)
            heights.append(h);margins.append(1000*s['minimum_native_joint_margin_rad']);closures.append(a['endpoints'][0]['joint_closure_rad'])
        axes[1].plot(heights,margins,'o-',color=color,label=label)
        axes[2].plot(heights,[float('nan') if x is None else x for x in closures],'o-',color=color,label=label)
    axes[0].set(xlabel='Executed simulation time (s)',ylabel='Actual rung offset (mm)',xlim=(0,125),yticks=[5,7.5,10,12.5,15],title='Strict result:3/5 to4/5')
    axes[0].legend(fontsize=8,loc='upper left');axes[1].axhline(0,color='black',ls=':',lw=1)
    axes[1].set(xlabel='Actual rung offset (mm)',ylabel='Runtime minimum margin (mrad)',title='Native joint bounds retained')
    axes[2].axhline(.04,color='black',ls=':',lw=1,label='0.04rad closure limit')
    axes[2].set(xlabel='Actual rung offset (mm)',ylabel='Maximum terminal closure error (rad)',title='Remaining15mm endpoint failure');axes[2].legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.12)
    fig.suptitle('Same horizon and endpoint: support retargeting repairs one known case')
    fig.supxlabel('Green circle:strict pass; red cross:failure. Missing closure points mean no endpoint reached.\nKnown-case zero-action calibration; not a learned-policy or held-out result.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(root/f'support_reference_comparison.{ext}',dpi=160)
    plt.close(fig)
    (root/'plot_provenance.json').write_text(json.dumps(dict(audits=sources,selection_sha256=sha(root/'selection.json'),source_sha256=sha(__file__),png_sha256=sha(root/'support_reference_comparison.png')),indent=2))


if __name__=='__main__':main()
