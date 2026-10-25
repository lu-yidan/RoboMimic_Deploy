"""Validate causal delivery, completed physics traces, and sparse mapping consistency."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.motion.reactive_realtime import SparseReference
p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('--baseline',type=Path);a=p.parse_args()
report=json.loads((a.job/'report.json').read_text());h=dict(np.load(a.job/'human.npz'));q=np.load(a.job/'trace.npz')['qpos']
assert report['completed'] and report['failure'] is None
assert len(q)==len(report['ticks'])
assert np.isfinite(q).all()
np.testing.assert_allclose(np.linalg.norm(q[:,3:7],axis=1),1,atol=1e-5)
for c in report['chunks']:
 assert c['observed_until']==c['end']-5
 assert c['received_s']>=c['source_observation_s']
 if c['first_use_s'] is not None:assert c['first_use_s']>=c['received_s']
# Independent direct formula, plus incremental vs whole-sequence resampling.
ws=Path(__file__).resolve().parents[3]
meta=json.loads((ws/'ScaleBFM/artifacts/scalebfm/humanoid_transformer_m/model_22200_tensorrt_metadata.json').read_text())
ik=json.loads((ws/'GMR/general_motion_retargeting/ik_configs/fbx_offline_to_g1.json').read_text())
a_ref=SparseReference(meta,ik);b_ref=SparseReference(meta,ik)
for end in range(4,len(h['positions'])+1,4):
 a_ref.append(dict(end=end,positions=h['positions'][end-4:end],rotations=h['rotations'][end-4:end]))
b_ref.append(dict(end=len(h['positions']),positions=h['positions'],rotations=h['rotations']))
np.testing.assert_allclose(a_ref.sample_pos,b_ref.sample_pos,atol=1e-6)
np.testing.assert_allclose(np.abs(np.sum(np.array(a_ref.sample_quat)*b_ref.sample_quat,axis=-1)),1,atol=1e-6)
pos=h['positions'][:,0];expected=np.stack([pos[:,0]*.9,(pos[:,8]-pos[:,0])*.8+pos[:,0]*.9,(pos[:,12]-pos[:,0])*.8+pos[:,0]*.9],axis=1)
np.testing.assert_allclose(a_ref.pos,expected,atol=1e-6)
assert [t['frame'] for t in report['ticks']]==list(range(len(q)))
print('PASS',a.job,'causal timestamps, finite physics, contiguous controls, incremental mapping/resampling')

if a.baseline:
    baseline=dict(np.load(a.baseline/'human.npz'));n=min(len(h['positions']),len(baseline['positions']))
    cut=int(report.get('intervention_s',2)*30);mode=report.get('opponent_mode','normal')
    np.testing.assert_allclose(h['positions'][:cut,0],baseline['positions'][:cut,0],atol=1e-6)
    indices=np.arange(n)
    if mode=='freeze':indices[cut:]=cut
    elif mode=='delay':indices[cut:]=np.maximum(cut,indices[cut:]-int(report['opponent_delay_ms']*.03))
    np.testing.assert_allclose(h['positions'][:n,1],baseline['positions'][indices,1],atol=1e-6)
    np.testing.assert_allclose(h['rotations'][:n,1],baseline['rotations'][indices,1],atol=1e-6)
    print('PASS intervention: unchanged prefix and correctly transformed opponent positions/rotations')
