"""Offline branch comparisons use real ONNX policies and never create transport."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import types

import numpy as np
import pytest

from tools.deployment.replay_branches import NoDDS, contract, load_input, make_diagnostic

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent/'RoboMimic_Deploy_under_table_review'


def write_changed(source,target,**updates):
    with np.load(source,allow_pickle=False) as archive:
        data = {name:archive[name].copy() for name in archive.files}
    data.update(updates)
    np.savez(target,**data)


def test_synthetic_input_has_explicit_provenance_and_strict_required_fields(tmp_path):
    source = tmp_path/'input.npz'
    make_diagnostic(source)
    data,origin = load_input(source,'PASSIVE')
    assert origin['kind'] == 'synthetic_diagnostic'
    assert data['q'].shape == (280,29)
    assert data['remote_raw'].dtype == np.uint8
    write_changed(source,tmp_path/'missing.npz',imu_gyro=np.full((280,3),np.nan))
    with pytest.raises(ValueError,match='Missing/nonfinite'):
        load_input(tmp_path/'missing.npz','PASSIVE')


def test_unknown_provenance_and_unsupported_policy_request_fail_closed(tmp_path):
    source = tmp_path/'input.npz'
    make_diagnostic(source)
    write_changed(source,tmp_path/'unknown.npz',provenance_json=json.dumps({'kind':'maybe_real'}))
    with pytest.raises(ValueError,match='provenance'):
        load_input(tmp_path/'unknown.npz','PASSIVE')
    with np.load(source) as archive:
        remote = archive['remote_raw'].copy()
    remote[0,2:4] = np.frombuffer(struct.pack('<H',1<<9),np.uint8)  # B selects excluded Loco.
    write_changed(source,tmp_path/'excluded.npz',remote_raw=remote)
    with pytest.raises(ValueError,match='outside'):
        load_input(tmp_path/'excluded.npz','PASSIVE')


def test_transport_symbols_raise_before_constructing_any_transport():
    guard = NoDDS()
    module = types.ModuleType('test_guard_dds')
    guard.exec_module(module)
    before = len(guard.attempts)
    with pytest.raises(RuntimeError,match='construction forbidden'):
        module.BridgeCmdPublisher()
    assert len(guard.attempts) == before+1
    with pytest.raises(ImportError,match='forbidden'):
        guard.find_spec('cyclonedds.domain')


def test_explicit_profile_requires_exact_model_hash(tmp_path):
    wrong = tmp_path/'wrong.onnx'
    wrong.write_bytes(b'not a model')
    with pytest.raises(ValueError,match='weight hash'):
        contract(ROOT,'amp','amp_0309_1',wrong)


def test_cli_preserves_input_and_existing_evidence_when_output_overlaps(tmp_path):
    source = tmp_path/'input.npz'
    make_diagnostic(source)
    before = source.read_bytes()
    result = subprocess.run([sys.executable,str(ROOT/'tools/deployment/replay_branches.py'),
        '--baseline-root',str(ROOT),'--input',str(source),'--initial-state','PASSIVE',
        '--amp-profile','amp_0309_1','--smp-profile','v34_balanced4_hard19000',
        '--amp-model',str(ROOT/'policy/amp/model/amp_0309_1.onnx'),
        '--smp-model',str(ROOT/'policy/smp_recovery/model/smp_v34_balanced4_hard19000_93d.onnx'),
        '--output-dir',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode == 2 and 'new or empty directory' in result.stderr
    assert source.read_bytes() == before
    assert not (tmp_path/'baseline_request.json').exists()


@pytest.mark.skipif(not BASELINE.exists(),reason='Detached under-the-table checkout unavailable')
@pytest.mark.parametrize('scenario',['nominal','mixed'])
def test_actual_runtime_fsm_and_models_compare_nominal_and_priority_paths(tmp_path,scenario):
    source,out = tmp_path/'input.npz',tmp_path/'result'
    make_diagnostic(source,scenario)
    subprocess.run([sys.executable,str(ROOT/'tools/deployment/replay_branches.py'),
        '--baseline-root',str(BASELINE),'--input',str(source),'--initial-state','PASSIVE',
        '--amp-profile','amp_0309_1','--smp-profile','v34_balanced4_hard19000',
        '--amp-model',str(ROOT/'policy/amp/model/amp_0309_1.onnx'),
        '--smp-model',str(ROOT/'policy/smp_recovery/model/smp_v34_balanced4_hard19000_93d.onnx'),
        '--output-dir',str(out)],check=True,capture_output=True,text=True)
    report = json.loads((out/'comparison.json').read_text())
    assert report['status'] == 'ready_for_real_log'
    assert report['hardware_baseline_verified'] is False
    if scenario == 'mixed':
        assert report['comparison']['first_divergence']['frame'] == 180  # F1 is now same-tick damping.
    else:
        assert report['comparison']['equal']
    baseline = json.loads((out/'baseline.json').read_text())
    candidate = json.loads((out/'candidate.json').read_text())
    assert not baseline['dds_construction_attempts'] and not candidate['dds_construction_attempts']
    for left,right in zip(baseline['records'][:180],candidate['records'][:180]):
        for key in ('qtarget','kp','kd'):
            np.testing.assert_allclose(left[key],right[key],atol=1e-6,rtol=0)
    if scenario == 'nominal':
        assert len(baseline['records']) == len(candidate['records']) == 320
        return
    for result in (baseline,candidate):
        assert len(result['records']) == 271  # Stop on SELECT; do not replay beyond outer-loop exit.
        stale = result['records'][250]
        assert stale['state_timeout'] and stale['request_damping'] and stale['executed_state']==-1
        np.testing.assert_equal(stale['kp'],np.zeros(29))
        np.testing.assert_equal(stale['kd'],np.full(29,8.))
    assert candidate['records'][180]['damping_output']
    assert not baseline['records'][180]['damping_output']
