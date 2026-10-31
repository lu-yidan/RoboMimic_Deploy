"""Compare two offline IK path constructions, preserving their failure labels."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    with np.load(root/'outputs/v2_swing_two_reference_20261029_v1/plan.npz') as z:
        original_time=z['time'];original_q=z['qpos']
    fig,axes=plt.subplots(2,2,figsize=(13,8.5))
    manifest={}
    for version,label,color in [('v1','Independent knots: rejected','#c44e52'),('v2','Continuation + 20 ms knots: kinematics passed','#0072b2')]:
        folder=root/f'outputs/wholebody_hand_reference_candidate_20261029_{version}'
        report=json.loads((folder/'summary.json').read_text())
        with np.load(folder/'candidate_knots.npz') as z:time=z['time'];q=z['qpos']
        parent=np.column_stack([np.interp(time,original_time,original_q[:,j]) for j in range(q.shape[1])])
        assert q.shape[1]==36, 'Figure assumes free root plus 29 scalar joints; review changed model'
        delta=q-parent;mask=time>=123.66
        axes[0,0].plot(time[mask],np.linalg.norm(delta[mask,:3],axis=1)*1000,color=color,label=label)
        axes[0,1].plot(time[mask],abs(delta[mask,7:]).max(axis=1)*1000,color=color)
        speed=abs(np.diff(delta[:,7:],axis=0)/np.diff(time)[:,None]).max(axis=1)
        selected=time[1:]>123.66
        axes[1,0].semilogy(time[1:][selected],np.maximum(speed[selected],1e-7),color=color)
        manifest[version]={'summary_sha256':hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest(),
            'knots_sha256':hashlib.sha256((folder/'candidate_knots.npz').read_bytes()).hexdigest(),
            'passed':report['passed']}
    axes[0,0].set(title='Root translation correction',ylabel='Magnitude (mm)')
    axes[0,0].axhline(3,color='gray',linestyle='--',linewidth=.8)
    axes[0,0].legend(fontsize=8,loc='upper left')
    axes[0,1].set(title='Largest joint correction',ylabel='Absolute correction (mrad)')
    axes[1,0].axhline(.1,color='gray',linestyle='--',linewidth=.8,label='Declared 0.1 rad/s limit')
    axes[1,0].set(title='Added reference joint speed (log scale)',ylabel='Maximum absolute difference (rad/s)')
    axes[1,0].legend(fontsize=9)
    for ax in axes.flat[:3]:
        ax.set_xlabel('Reference time (s)');ax.grid(alpha=.2)
    axes[1,1].axis('off')
    axes[1,1].text(.02,.97,
        'Fixed +2 mm X / +2 mm Z target shift\n\n'
        'Independent knots: rejected\n'
        '  Dense position error: 0.370 mm\n  Added joint speed: 0.898 rad/s\n\n'
        'Continuation + denser knots: kinematics passed\n'
        '  Dense position error: 0.000205 mm\n  Added joint speed: 0.01523 rad/s\n\n'
        'Both: zero flagged collision samples at 5 ms\n'
        'Original controller and official v2 unchanged\n'
        'No dynamics or camera accuracy claim',
        transform=axes[1,1].transAxes,va='top',fontsize=11,linespacing=1.5)
    fig.suptitle('Hand-target reference prototype: discrete feasibility is not path feasibility',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96))
    for ext in ('png','pdf'):fig.savefig(a.output/f'wholebody_reference_audit.{ext}',dpi=160)
    plt.close(fig)
    manifest['plot_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
