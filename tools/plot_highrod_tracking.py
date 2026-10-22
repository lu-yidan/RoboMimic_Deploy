"""Plot saved tracking errors without extrapolating failed trajectories."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=Path('outputs/highrod_height_mismatch_v1');data=json.loads((p/'margin_traces.json').read_text())
    # The state coordinate is resolved from the saved joint diagnostic observation.
    manifest=json.loads((p/'true_height_10mm/manifest.json').read_text())
    with np.load(manifest['reference']) as z:
        times=z['time'];q=z['qpos']
    with np.load(p/'true_height_10mm/trajectory.npz') as z:
        obs=z['observations'];actual=z['qpos'];t=z['time_s']
    # G1 free base has seven position coordinates followed by the29 motor joints.
    # Validate index against the separately exported ankle trace before plotting.
    trace=next(r for r in data['rows'] if r['true_height_mm']==10.)
    mask=(t>=45)&(t<=65)
    matches=[i for i in range(actual.shape[1]) if np.array_equal(actual[mask,i],trace['right_ankle_pitch_rad'])]
    assert len(matches)==1
    coordinate=matches[0]
    fig,axes=plt.subplots(2,1,figsize=(9,6.7),sharex=True,layout='constrained')
    window=(times>=52)&(times<=65)
    axes[0].plot(times[window],q[window,coordinate],color='black',lw=2,label='Common reference')
    for r in data['rows']:
        ts=np.asarray(r['time_s']);pos=np.asarray(r['right_ankle_pitch_rad']);ref=np.interp(ts,times,q[:,coordinate])
        label=f"Actual +{r['true_height_mm']:g} mm"
        axes[0].plot(ts,pos,label=label,ls='-' if r['accepted'] else '--')
        axes[1].plot(ts,1000*(pos-ref),ls='-' if r['accepted'] else '--')
    axes[0].axhline(trace['lower_limit_rad'],color='black',ls=':',label='Native lower limit')
    axes[1].axhline(0,color='black',ls=':',lw=.8)
    axes[0].set_ylabel('Right ankle pitch (rad)');axes[1].set_ylabel('Actual minus reference (mrad)')
    axes[1].set_xlabel('Simulation time (s)');axes[1].set_xlim(52,65)
    axes[0].legend(fontsize=8,ncol=2)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.suptitle('High-rod failure: native limit reached despite reference headroom')
    fig.supxlabel('Same reference, zero actions, -3 deg normal bias. Traces end at failure.\nSaved states at50Hz; safety checks at2000Hz. No causal claim from these curves alone.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(p/f'tracking_error.{ext}',dpi=160)
    plt.close(fig)
    sha=lambda file:hashlib.sha256(Path(file).read_bytes()).hexdigest()
    (p/'tracking_plot_provenance.json').write_text(json.dumps(dict(source_sha256=sha(__file__),data_sha256=sha(p/'margin_traces.json'),
        reference_sha256=sha(manifest['reference']),png_sha256=sha(p/'tracking_error.png'),resolved_qpos_coordinate=coordinate),indent=2))


if __name__=='__main__':main()
