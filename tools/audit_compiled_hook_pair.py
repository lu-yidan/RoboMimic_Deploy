"""Audit the explicitly compiled scene, frozen controller and full-rate record."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from training.compiled_rung_geometry import CompiledPhysicalV2CycleEnv
from tools import audit_v2_multicycle as physical_audit
from tools.validate_swing_reference import ROOT, read, write, sha


def audit(out):
    out=out.resolve();m=read(out/'manifest.json');parent=Path(m['parent_experiment'])
    pm=read(parent/'manifest.json');chain=read(parent/'compiled_reference_chain_audit.json')
    assert m['cycles']==2 and m['scene_compiler']=='complete_mjspec_model_at_reset_v1'
    assert chain['passed'] and sha(parent/'compiled_reference_chain_audit.json')==m['compiled_parent_audit_sha256']
    assert m['hook_protocol_sha256']==sha(parent/'hook_uncertainty_protocol.json')
    assert m['offsets']==pm['evaluation_cases'][m['case']]
    assert m['arm'] in ('adopted_v2','contact_confirmed')
    assert (m['arm']=='contact_confirmed')==bool(m.get('contact_admission'))
    scene=read(out/'scene_variant.json')
    e=CompiledPhysicalV2CycleEnv(plan=m['plan_path'],cases={m['case']:m['offsets']},
        normal_bias_y_deg=pm['normal_bias_y_deg'],memory_enabled=pm['force_history_memory'])
    try:
        e.reset(options={'case':m['case']})
        geometry=e.variant['geometry_sha256']==scene['geometry_sha256']==m['scene_geometry_sha256']
    finally:e.close()
    original=physical_audit.PhysicalV2CycleEnv
    physical_audit.PhysicalV2CycleEnv=CompiledPhysicalV2CycleEnv
    try:task=physical_audit.audit(out)
    finally:physical_audit.PhysicalV2CycleEnv=original
    record=read(out/'physics_telemetry_record.json')
    assert record['manifest_sha256']==sha(out/'manifest.json')
    assert record['telemetry_sha256']==sha(out/'physics_telemetry.npz')
    assert record['trajectory_sha256']==sha(out/'trajectory.npz')
    with np.load(out/'physics_telemetry.npz') as f:z={k:f[k] for k in f.files}
    with np.load(out/'trajectory.npz') as f:x={k:f[k] for k in f.files}
    t=z['time_s'];indices=np.rint(x['time_s'][1:]/.0005).astype(int)-1
    checks=dict(compiled_scene=geometry,
        finite=all(bool(np.isfinite(v).all()) for v in z.values()),
        samples=len(t)==record['physics_samples']==round(float(x['time_s'][-1])/.0005),
        cadence=bool(len(t)>0 and abs(t[0]-.0005)<1e-10 and np.all(abs(np.diff(t)-.0005)<1e-9)),
        time_alignment=bool(np.all(abs(t[indices]-x['time_s'][1:])<1e-9)),
        velocity_alignment=bool(np.array_equal(z['post_velocity'][indices],x['qvel'][1:,:6])),
        torque_alignment=bool(np.array_equal(z['motor_torque_nm'][indices],x['torques'])),
        acceleration_alignment=bool(np.array_equal(z['solver_qacc'][indices],x['qacc_50hz'])))
    admission=None
    if m['arm']=='contact_confirmed':
        subprocess.run([sys.executable,'-m','tools.audit_hook_admission',str(out)],cwd=ROOT,check=True)
        admission=read(out/'hook_admission_audit.json')
    report=dict(passed=all(checks.values()) and task['passed'] and (admission is None or admission['passed']),
        integrity_passed=all(checks.values()) and task['integrity_passed'],checks=checks,
        physical_policy_passed=task['passed'],contact_confirmation_passed=None if admission is None else admission['trace_checks_passed'],
        case=m['case'],arm=m['arm'],manifest_sha256=sha(out/'manifest.json'),
        trajectory_sha256=sha(out/'trajectory.npz'),telemetry_sha256=sha(out/'physics_telemetry.npz'),
        physical_policy_audit_sha256=sha(out/'v2_multicycle_audit.json'),
        hook_audit_sha256=sha(out/'hook_admission_audit.json') if admission is not None else None,
        scene_geometry_sha256=scene['geometry_sha256'],source_sha256=sha(__file__),
        scope='Complete compiled-scene binding; independent saved-state, controller and policy reconstruction; full-rate runtime telemetry alignment. Physical failure remains failure. This is not a separate full dynamics reproduction or baseline promotion.')
    write(out/'compiled_pair_audit.json',report)
    (out/'compiled_pair_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    audit(p.parse_args().output)
