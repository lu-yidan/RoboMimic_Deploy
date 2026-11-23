"""Recorded standing qualification and explicit missing-state assumptions."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from tools.acp.evaluate_recorded_entry import initialize_recorded_pose,foot_sample,standing_windows
from tools.acp.view_recorded_entry import load_trace


def fixture_log(folder,change=None):
    n=75
    values=dict(q=np.zeros((n,29)),dq=np.zeros((n,29)),
        pelvis_quat_w=np.tile([1.,0.,0.,0.],(n,1)),ang_vel=np.zeros((n,3)),
        vel_cmd=np.zeros((n,3)),executed_fsm_state=np.full((n,1),16),time_s=np.arange(n)[:,None]*.02)
    if change:change(values)
    fields={k:v.shape[1] for k,v in values.items()}
    path=folder/'fixture.json'
    path.write_text(json.dumps(dict(dtype='float32',record_dim=sum(fields.values()),fields=fields,
        git_commit='6a25875-test-only',test_only=True)))
    np.concatenate(list(values.values()),axis=1).astype(np.float32).tofile(path.with_suffix('.bin'))
    return path


def test_qualified_standing_requires_continuous_quiet_records(tmp_path):
    assert standing_windows(fixture_log(tmp_path))[0]['frames']==75
    for change in (lambda d:d['vel_cmd'].__setitem__((slice(None),0),.1),
                   lambda d:d['ang_vel'].__setitem__((slice(None),0),.2),
                   lambda d:d['time_s'].__setitem__((slice(38,None),0),d['time_s'][38:,0]+2.)):
        assert not standing_windows(fixture_log(tmp_path,change))


def test_height_fitting_preserves_every_recorded_joint_and_imu_value():
    root=Path(__file__).resolve().parents[1]
    model=mujoco.MjModel.from_xml_path(str(root/'g1_description/scene.xml'))
    data=mujoco.MjData(model)
    q=np.zeros(29);q[[0,6]]=-.3;q[[3,9]]=.62;q[[4,10]]=-.33
    quat=np.array([.99,.02,-.03,.1]);quat/=np.linalg.norm(quat)
    pose=dict(q=q,dq=np.linspace(-.01,.01,29),pelvis_quat_w=quat,ang_vel=np.array([.02,-.01,.03]))
    initial,feet=initialize_recorded_pose(model,data,pose)
    np.testing.assert_array_equal(data.qpos[7:],pose['q'])
    np.testing.assert_array_equal(data.qvel[6:],pose['dq'])
    np.testing.assert_allclose(data.qpos[3:7],quat,atol=1e-14)
    np.testing.assert_array_equal(data.qvel[3:6],pose['ang_vel'])
    np.testing.assert_array_equal(data.qvel[:3],0.)
    loads,heights=foot_sample(model,data,feet)
    assert heights.min()==pytest.approx(.002,abs=1e-9)


def test_viewer_rejects_a_changed_trace(tmp_path):
    path=tmp_path/'pose.npz'
    np.savez(path,qpos=np.zeros((2,36)),time_s=[0.,.02])
    row=dict(trace_file=path.name,trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (tmp_path/'results.json').write_text(json.dumps(dict(results=[row])))
    assert len(load_trace(path)[1]['time_s'])==2
    np.savez(path,qpos=np.ones((2,36)),time_s=[0.,.02])
    with pytest.raises(ValueError,match='hash mismatch'):load_trace(path)
