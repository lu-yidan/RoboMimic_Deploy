"""Visualize audited desired and realized joint acceleration."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=Path('outputs/highrod_dynamics_trace_v1');d=json.loads((p/'diagnostic.json').read_text())
    assert d['passed'] and all(d['all_trajectory_arrays_exact'].values())
    source=p/'true_height_12.5mm/dynamics.npz'
    sha=lambda file:hashlib.sha256(Path(file).read_bytes()).hexdigest()
    assert sha(source)==d['dynamics_sha256']
    with np.load(source) as z:
        t=z['time_s'];i=d['motor_index'];target=z['desired_acceleration'][:,i];actual=z['average_acceleration'][:,i]
    keep=t>=59.;t=t[keep];target=target[keep];actual=actual[keep]
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    ax.plot(t,target,lw=1.2,label='Controller target',color='#234d82')
    ax.plot(t,actual,lw=1.,label='Realized interval average',color='#c25e29')
    ax.axhline(0,color='black',lw=.7);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    ax.set(xlim=(59,float(t[-1])),xlabel='Simulation time (s)',ylabel='Right ankle acceleration (rad/s²)')
    ax.legend();fig.suptitle('Contact-constrained motion differs from the prescribed acceleration')
    fig.supxlabel('Actual +12.5mm rod; original failed trajectory reproduced exactly.\n200Hz control records. Difference alone does not identify a causal mechanism.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(p/f'acceleration_tracking.{ext}',dpi=160)
    plt.close(fig)
    (p/'plot_provenance.json').write_text(json.dumps(dict(source_sha256=sha(__file__),diagnostic_sha256=sha(p/'diagnostic.json'),png_sha256=sha(p/'acceleration_tracking.png')),indent=2))


if __name__=='__main__':main()
