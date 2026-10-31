"""Independent timing/cap and paired-array audit of the open-loop pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.validate_swing_reference import read,write,sha


def audit_timing(out):
    out=out.resolve()
    m=read(out/'manifest.json');protocol=read(out/'timed_loading_protocol.json')
    record=read(out/'timed_loading_record.json');task=read(out/'v2_multicycle_audit.json')
    assert protocol==m['timed_loading'] and not protocol['feedback_used'] and not m.get('contact_admission')
    assert sha(out/'timed_loading_protocol.json')==m['timed_loading_protocol_sha256']==record['protocol_sha256']
    assert sha(out/'physics_telemetry.npz')==record['telemetry_sha256']
    assert sha(out/'timed_loading_trace.json')==record['trace_sha256']
    assert sha(out/'trajectory.npz')==record['trajectory_sha256']==task['trajectory_sha256']
    assert task['manifest_sha256']==sha(out/'manifest.json')
    with np.load(out/'physics_telemetry.npz') as f:z={k:f[k] for k in f.files}
    with np.load(out/'trajectory.npz') as f:x={k:f[k] for k in f.files}
    with np.load(out/'reference.npz') as f:ref={k:f[k] for k in f.files}
    t=z['time_s'];clock=np.r_[0.,t[9:-1:10]];ends=ref['cycle_ends_s']
    positions=np.clip(np.searchsorted(ref['time'],clock,side='right')-1,0,len(ref['time'])-2)+1
    active=ref['active'][positions].astype(bool)
    expected={}
    for group,column in [('left_hook',2),('right_hook',3)]:
        for i in np.flatnonzero(active[1:,column]&~active[:-1,column])+1:
            if clock[i]<=ends[0]+1e-7:continue
            start=float(clock[i]);j=int(np.searchsorted(clock,start+protocol['delays_seconds'][group]-1e-8))
            expected[(group,start)]=float(clock[j]) if j<len(clock) else None
    seen=set();error=0.;timing=True;no_feedback=True
    for row in read(out/'timed_loading_trace.json'):
        when=row['time_s'];caps=np.full(4,350.)
        no_feedback=no_feedback and row['controller_type']=='timed_open_loop' and 'measurements' not in row
        for group,hook in row['hooks'].items():
            key=(group,hook['start']);seen.add(key);assert key in expected
            release=expected[key]
            expected_release=None if release is None or when<release else release
            timing=timing and hook['released']==expected_release
            cap=protocol['probe_force_n']/2
            if expected_release is not None:
                u=float(np.clip((when-release)/protocol['ramp_seconds'],0,1));w=u*u*(3-2*u)
                cap=(protocol['probe_force_n']+(700-protocol['probe_force_n'])*w)/2
            caps[2 if group=='left_hook' else 3]=cap
        error=max(error,float(abs(caps-row['caps_per_contact_n']).max()))
    indices=np.rint(x['time_s'][1:]/.0005).astype(int)-1
    checks=dict(timed_events_exact=seen==set(expected),release_on_declared_clock=timing,
        cap_reconstruction=error<1e-10,no_feedback_trace=no_feedback,
        finite=all(bool(np.isfinite(v).all()) for v in z.values()),
        cadence=bool(len(t)>0 and abs(t[0]-.0005)<1e-10 and np.all(abs(np.diff(t)-.0005)<1e-9)),
        complete=len(t)==record['physics_samples']==round(float(x['time_s'][-1])/.0005),
        time_alignment=bool(np.all(abs(t[indices]-x['time_s'][1:])<1e-9)),
        velocity_alignment=bool(np.array_equal(z['post_velocity'][indices],x['qvel'][1:,:6])),
        torque_alignment=bool(np.array_equal(z['motor_torque_nm'][indices],x['torques'])),
        acceleration_alignment=bool(np.array_equal(z['solver_qacc'][indices],x['qacc_50hz'])))
    report=dict(passed=all(checks.values()) and task['passed'],checks=checks,
        task_passed=task['passed'],cap_reconstruction_error=error,
        expected_events=[dict(group=g,start_s=t,release_s=r) for (g,t),r in expected.items()],
        manifest_sha256=sha(out/'manifest.json'),trajectory_sha256=record['trajectory_sha256'],
        telemetry_sha256=record['telemetry_sha256'],trace_sha256=record['trace_sha256'],
        task_audit_sha256=sha(out/'v2_multicycle_audit.json'),source_sha256=sha(__file__),
        scope='Fresh deterministic open-loop dynamics with timing and telemetry consistency checks. The timed trigger has no contact confirmation; task failure remains failure. No automatic adoption.')
    target=out/'timed_loading_audit.json';assert not target.exists()
    write(target,report);(out/'timed_loading_audit_source.py').write_bytes(Path(__file__).read_bytes())
    return report


def main():
    from training.compiled_rung_geometry import CompiledPhysicalV2CycleEnv
    from tools import audit_v2_multicycle as physical
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);out=p.parse_args().run.resolve()
    m=read(out/'manifest.json');parent=Path(m['parent_experiment']);pm=read(parent/'manifest.json')
    assert m['cycles']==2 and m['arm']=='timed_open_loop' and not m.get('contact_admission')
    assert m['scene_compiler']=='complete_mjspec_model_at_reset_v1'
    assert sha(parent/'compiled_reference_chain_audit.json')==m['compiled_parent_audit_sha256']
    assert read(parent/'compiled_reference_chain_audit.json')['passed']
    assert m['offsets']==pm['evaluation_cases'][m['case']]
    assert sha(parent/'hook_uncertainty_protocol.json')==m['hook_protocol_sha256']
    protocol=m['timed_loading'];declared=Path(protocol['grid_protocol'])
    assert sha(declared)==protocol['grid_protocol_sha256']
    d=read(declared)
    assert d['cases']==pm['evaluation_cases'] and d['single_manifest_sha256']==sha(parent/'manifest.json')
    assert all(protocol[k]==d[k] for k in ('delays_seconds','probe_force_n','ramp_seconds'))
    for relative,digest in d['sources'].items():
        from tools.validate_swing_reference import ROOT
        assert sha(ROOT/relative)==digest
    e=CompiledPhysicalV2CycleEnv(plan=m['plan_path'],cases={m['case']:m['offsets']},
        normal_bias_y_deg=pm['normal_bias_y_deg'],memory_enabled=pm['force_history_memory'])
    try:
        e.reset(options={'case':m['case']})
        assert e.variant['geometry_sha256']==m['scene_geometry_sha256']==read(out/'scene_variant.json')['geometry_sha256']
    finally:e.close()
    old=physical.PhysicalV2CycleEnv;physical.PhysicalV2CycleEnv=CompiledPhysicalV2CycleEnv
    try:task=physical.audit(out)
    finally:physical.PhysicalV2CycleEnv=old
    timing=audit_timing(out)
    report=dict(passed=task['passed'] and timing['passed'],
        integrity_passed=task['integrity_passed'] and all(timing['checks'].values()),
        physical_policy_passed=task['passed'],timing_passed=all(timing['checks'].values()),
        case=m['case'],arm=m['arm'],manifest_sha256=sha(out/'manifest.json'),
        trajectory_sha256=sha(out/'trajectory.npz'),telemetry_sha256=sha(out/'physics_telemetry.npz'),
        physical_policy_audit_sha256=sha(out/'v2_multicycle_audit.json'),
        timing_audit_sha256=sha(out/'timed_loading_audit.json'),source_sha256=sha(__file__),
        scene_geometry_sha256=m['scene_geometry_sha256'],automatic_promotion=False)
    target=out/'compiled_pair_audit.json';assert not target.exists();write(target,report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
