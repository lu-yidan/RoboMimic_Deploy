"""Exact replay of an audited v2 run with existing read-only 2000 Hz telemetry.

Solver-stage accelerations/forces and post-integration velocities are labelled
separately. Base acceleration is not whole-robot COM acceleration.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from tools import evaluate_v2_multicycle as evaluator
from tools.validate_swing_reference import read,write,sha,ROOT
from training.physics_telemetry import install_physics_telemetry


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.run.resolve();m=read(source/'manifest.json')
    if any(m.get(key) for key in ('contact_admission','timed_loading','scene_compiler')):
        raise ValueError('This baseline telemetry entry cannot reproduce experimental controllers or compiled scenes')
    gate=read(source/'v2_multicycle_audit.json')
    assert gate['passed'] and gate['manifest_sha256']==sha(source/'manifest.json')
    assert gate['trajectory_sha256']==sha(source/'trajectory.npz')
    assert gate['independent_audit_sha256']==sha(source/'independent_audit.json')
    traces=[];original_class=evaluator.PhysicalV2CycleEnv
    class TelemetryEnv(original_class):
        def reset(self,**kwargs):
            observation=super().reset(**kwargs)
            traces.append(install_physics_telemetry(self))
            return observation
    evaluator.PhysicalV2CycleEnv=TelemetryEnv
    old_argv=sys.argv
    sys.argv=['evaluate_v2_multicycle','--candidate',m['parent_experiment'],'--plan',m['plan_path'],
        '--case',m['case'],'--output',str(a.output.resolve())]
    if m.get('two_cycle_gate'):sys.argv+=['--two-cycle-gate',m['two_cycle_gate']]
    try:evaluator.main()
    finally:
        evaluator.PhysicalV2CycleEnv=original_class;sys.argv=old_argv
    assert len(traces)==1
    trace=traces[0];data={k:v[:trace['count']] for k,v in trace['data'].items()}
    np.savez_compressed(a.output/'physics_telemetry.npz',**data)
    with np.load(source/'trajectory.npz') as old,np.load(a.output/'trajectory.npz') as new:
        exact={k:bool(np.array_equal(old[k],new[k])) for k in old.files}
        assert set(old.files)==set(new.files)
        indices=np.rint(new['time_s'][1:]/.0005).astype(int)-1
        checks=dict(exact_complete_trajectory=all(exact.values()),
            finite=all(bool(np.isfinite(v).all()) for v in data.values()),
            start=abs(data['time_s'][0]-.0005)<1e-9,
            cadence=bool(np.all(abs(np.diff(data['time_s'])-.0005)<1e-9)),
            complete=abs(data['time_s'][-1]-new['time_s'][-1])<1e-9,
            sample_count=len(data['time_s'])==int(round(new['time_s'][-1]/.0005)),
            time_alignment=bool(np.all(abs(data['time_s'][indices]-new['time_s'][1:])<1e-9)),
            velocity_alignment=bool(np.array_equal(data['post_velocity'][indices],new['qvel'][1:,:6])),
            torque_alignment=bool(np.array_equal(data['motor_torque_nm'][indices],new['torques'])),
            acceleration_alignment=bool(np.array_equal(data['solver_qacc'][indices],new['qacc_50hz'])))
    # Numeric reporting only; these are not new acceptance thresholds.
    t=data['time_s'];acc=data['solver_qacc'];norm=np.linalg.norm(acc[:,:3],axis=1)
    force=data['group_force_world_n'];ref=np.load(m['plan_path']);cycles=[]
    for i,end in enumerate(ref['cycle_ends_s']):
        start=0. if i==0 else ref['cycle_ends_s'][i-1]
        mask=(t>start+1e-7)&(t<=end+1e-7);tail=(t>end-2)&(t<=end+1e-7)
        peak=np.flatnonzero(mask)[int(np.argmax(norm[mask]))]
        cycles.append(dict(cycle=i+1,peak_root_acceleration_m_s2=float(norm[peak]),peak_root_acceleration_time_s=float(t[peak]),
            peak_joint_acceleration_rad_s2=float(abs(acc[mask,6:]).max()),
            peak_group_force_norm_n={g:float(np.linalg.norm(force[mask,j,:],axis=1).max()) for j,g in enumerate(trace['groups'])},
            minimum_endpoint_vertical_force_n={g:float(force[tail,j,2].min()) for j,g in enumerate(trace['groups'])}))
    report=dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},exact_arrays=exact,
        source_run=str(source),source_audit_sha256=sha(source/'v2_multicycle_audit.json'),
        source_trajectory_sha256=sha(source/'trajectory.npz'),trajectory_sha256=sha(a.output/'trajectory.npz'),
        telemetry_sha256=sha(a.output/'physics_telemetry.npz'),groups=trace['groups'],physics_samples=trace['count'],
        cycles=cycles,source_sha256=sha(__file__),
        scope='Read-only 2000 Hz telemetry in a fresh continuous run; every saved trajectory array must reproduce the previously audited run exactly. Solver-stage accelerations/forces, post-integration timestamps and velocities. Root acceleration is not COM acceleration. Diagnostic peaks do not establish hardware safety or add a new success gate.')
    write(a.output/'telemetry_audit.json',report);(a.output/'telemetry_source.py').write_bytes(Path(__file__).read_bytes())
    assert report['passed']
    subprocess.run([sys.executable,'-m','tools.audit_v2_multicycle',str(a.output.resolve())],cwd=ROOT,check=True)
    print(json.dumps(dict(telemetry_passed=report['passed'],samples=trace['count'],cycles=cycles)),flush=True)


if __name__=='__main__':main()
