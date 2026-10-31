"""Plot audited confirmation delays against the declared fixed schedule."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--timed-protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.analysis.read_text());timed=json.loads(a.timed_protocol.read_text())
    assert data['declared_cases']==16 and len(data['cases'])==16
    delays=timed['delays_seconds']
    grids={hand:np.full((4,4),np.nan) for hand in ('right_hook','left_hook')}
    unavailable=[];xs={};zs={}
    for case in data['cases']:
        match=re.fullmatch(r'hook_grid_x(\d)_z(\d)',case['case']);assert match
        i,j=map(int,match.groups());offset=case['offsets']['rung_06'];xs[i]=offset[0]*1000;zs[j]=offset[2]*1000
        result=case['arms']['contact_confirmed']['complete_cycle_metrics']
        if result is None:
            unavailable.append(case['case']);continue
        for hand in grids:
            events=[e for e in result['events'] if e['group']==hand]
            assert len(events)==1
            grids[hand][j,i]=events[0]['delay_s']*1000
    difference={hand:g-1000*delays[hand] for hand,g in grids.items()}
    finite=np.concatenate([x[np.isfinite(x)] for x in difference.values()])
    scale=max(5.,float(abs(finite).max())) if len(finite) else 5.
    fig,axes=plt.subplots(1,2,figsize=(12,5.8))
    colormap=plt.get_cmap('coolwarm').copy();colormap.set_bad('#dddddd')
    for ax,hand in zip(axes,grids):
        picture=ax.imshow(difference[hand],origin='lower',cmap=colormap,vmin=-scale,vmax=scale)
        for j in range(4):
            for i in range(4):
                value=grids[hand][j,i]
                text='unavailable' if not np.isfinite(value) else f'{value:.0f} ms\n({difference[hand][j,i]:+.0f})'
                ax.text(i,j,text,ha='center',va='center',fontsize=10)
        ax.set_xticks(range(4),[f'{xs[i]:+.2f}' for i in range(4)])
        ax.set_yticks(range(4),[f'{zs[j]:+.2f}' for j in range(4)])
        ax.set_xlabel('Actual rod X offset (mm)');ax.set_ylabel('Actual rod Z offset (mm)')
        ax.set_title(f"{hand.replace('_',' ').title()} | fixed schedule {delays[hand]*1000:.0f} ms")
    fig.subplots_adjust(top=.84,bottom=.19,left=.07,right=.88,wspace=.28)
    bar=fig.colorbar(picture,cax=fig.add_axes([.91,.24,.02,.52]));bar.set_label('Confirmed minus fixed delay (ms)')
    fig.suptitle('Contact-confirmed timing varies with rod position',fontsize=16)
    fig.text(.07,.065,'Cells: audited confirmation delay; parentheses: difference from the declared fixed schedule.\n'
        'Unavailable records are not filled with zero. Timing adaptation alone does not establish better physical performance.',fontsize=10)
    a.output.mkdir(parents=True,exist_ok=False)
    for ext in ('png','pdf'):fig.savefig(a.output/f'confirmation_delays.{ext}',dpi=160)
    plt.close(fig)
    report=dict(analysis_sha256=hashlib.sha256(a.analysis.read_bytes()).hexdigest(),
        timed_protocol_sha256=hashlib.sha256(a.timed_protocol.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),unavailable_cases=unavailable,
        fixed_delays_s=delays,scope='Complete integrity-audited contact traces only; schedule difference, not performance superiority.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
