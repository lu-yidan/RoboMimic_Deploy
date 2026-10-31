"""Check probing trace consistency and full-rate telemetry, separately from task success."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from tools.validate_swing_reference import read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    p.add_argument('--report-name',default='hook_admission_audit.json')
    a=p.parse_args();out=a.run.resolve()
    if Path(a.report_name).name!=a.report_name or not a.report_name.endswith('.json'):
        raise ValueError('Report name must be a local JSON filename')
    report_path=out/a.report_name
    if report_path.exists():raise FileExistsError('Preserve the previous audit; choose a new report name')
    m=read(out/'manifest.json');protocol=read(out/'hook_admission_protocol.json');record=read(out/'hook_admission_record.json')
    assert m['contact_admission']==protocol and sha(out/'hook_admission_protocol.json')==m['hook_admission_protocol_sha256']==record['protocol_sha256']
    assert sha(out/'physics_telemetry.npz')==record['telemetry_sha256'] and sha(out/'hook_admission_trace.json')==record['trace_sha256']
    assert sha(out/'trajectory.npz')==record['trajectory_sha256']
    cfg=protocol['parameters'];trace=read(out/'hook_admission_trace.json')
    z=dict(np.load(out/'physics_telemetry.npz'));x=dict(np.load(out/'trajectory.npz'));t=z['time_s']
    index=np.rint(x['time_s'][1:]/.0005).astype(int)-1
    checks=dict(finite=all(bool(np.isfinite(v).all()) for v in z.values()),nonempty_probe_trace=bool(trace),
        telemetry_complete=len(t)==record['physics_samples']==round(float(x['time_s'][-1])/.0005),
        cadence=bool(np.all(abs(np.diff(t)-.0005)<1e-9)),
        time_alignment=bool(np.all(abs(t[index]-x['time_s'][1:])<1e-9)),
        velocity_alignment=bool(np.array_equal(z['post_velocity'][index],x['qvel'][1:,:6])),
        torque_alignment=bool(np.array_equal(z['motor_torque_nm'][index],x['torques'])),
        acceleration_alignment=bool(np.array_equal(z['solver_qacc'][index],x['qacc_50hz'])))
    env=WallSupportEnv(Path(m['bundle']));limits=env.limits.copy();env.close()
    ref=dict(np.load(out/'reference.npz'));ends=ref['cycle_ends_s']
    events={};cap_error=vertical_error=force_violation=torque_violation=0.;first_end=float(ends[0])
    for row in trace:
        when=row['time_s'];assert when>first_end
        caps=np.full(4,350.)
        for group,hook in row['hooks'].items():
            k=2 if group=='left_hook' else 3
            # A previously admitted left hook remains in state when the next
            # cycle's right hook starts probing. Its original start time, not
            # the cycle of this later trace row, identifies the same event.
            event_cycle=int(np.searchsorted(ends,hook['start'],side='right'))+1
            assert event_cycle<=row['cycle'] and hook['start']<=when
            key=(event_cycle,group,hook['start']);events.setdefault(key,[]).append(row)
            if hook['confirmed'] is None:caps[k]=cfg['probe_force_n']/2.
            else:
                u=float(np.clip((when-hook['confirmed'])/cfg['ramp_seconds'],0,1));w=u*u*(3-2*u)
                caps[k]=(cfg['probe_force_n']+(700.-cfg['probe_force_n'])*w)/2.
            if group in row['measurements']:
                sample=round(when/.0005)-1;j=record['groups'].index(group)
                vertical_error=max(vertical_error,abs(row['measurements'][group]['vertical_n']-z['group_force_world_n'][sample,j,2]))
        cap_error=max(cap_error,float(abs(caps-row['caps_per_contact_n']).max()))
        forces=np.asarray(row['force_plan']);upper=np.repeat(caps[np.asarray(row['active'],bool)],2)
        assert forces.shape==(len(upper),3)
        force_violation=max(force_violation,float(np.max(forces[:,2]-upper)),float(np.max(-forces[:,2])))
        for angle in np.arange(8)*np.pi/4+np.pi/8:
            force_violation=max(force_violation,float(np.max(forces@np.array([np.cos(angle),np.sin(angle),-.7*np.cos(np.pi/8)]))))
        torques=np.asarray(row['torque_nm']);torque_violation=max(torque_violation,float(np.max(torques-limits[:,1])),float(np.max(limits[:,0]-torques)))
        assert row['base_equilibrium_residual']<=1e-3
    event_rows=[]
    for (cycle,group,start),rows in events.items():
        confirmed=next((r['hooks'][group]['confirmed'] for r in rows if r['hooks'][group]['confirmed'] is not None),None)
        window=[] if confirmed is None else [r for r in rows if confirmed-cfg['confirm_seconds']-1e-8<=r['time_s']<=confirmed+1e-8]
        good=bool(confirmed is not None and confirmed-start<=cfg['timeout_seconds']+1e-8 and len(window)>=round(cfg['confirm_seconds']/.005)+1)
        if good:
            good=all(group in r['measurements'] and r['measurements'][group]['intended'] and min(r['measurements'][group]['normal_n'],r['measurements'][group]['vertical_n'])>cfg['confirm_force_n'] for r in window)
            good=good and bool(np.all(abs(np.diff([r['time_s'] for r in window])-.005)<1e-8))
        event_rows.append(dict(cycle=cycle,group=group,start_s=start,confirmed_s=confirmed,confirmed_with_continuous_feedback=good))
    # Independently recover every planned hook activation on the actual 200 Hz
    # command clock. This guards against fixing duplication by silently
    # dropping a genuinely missing activation.
    command_times=np.r_[0.,t[9:-1:10]]
    reference_indices=np.clip(np.searchsorted(ref['time'],command_times,side='right')-1,0,len(ref['time'])-2)+1
    active=ref['active'][reference_indices].astype(bool)
    expected=[]
    for group,column in [('left_hook',2),('right_hook',3)]:
        onsets=np.flatnonzero(active[1:,column]&~active[:-1,column])+1
        for i in onsets:
            when=float(command_times[i])
            if when>first_end+1e-7:
                expected.append(dict(cycle=int(np.searchsorted(ends,when,side='right'))+1,group=group,start_s=when))
    expected.sort(key=lambda r:(r['start_s'],r['group']))
    ordered=sorted(event_rows,key=lambda r:(r['start_s'],r['group']))
    expected_exact=len(expected)==len(ordered) and all(
        a['cycle']==b['cycle'] and a['group']==b['group'] and abs(a['start_s']-b['start_s'])<1e-8
        for a,b in zip(expected,ordered))
    checks.update(cap_curve=cap_error<1e-10,vertical_sensor_matches_physics=vertical_error<1e-8,
        qp_vertical_force_and_friction_bounds=force_violation<=1e-3,native_motor_bounds=torque_violation<=1e-8,
        exact_reference_activation_events=expected_exact,
        all_hooks_confirmed=len(event_rows)==2*(m['cycles']-1) and all(r['confirmed_with_continuous_feedback'] for r in event_rows))
    task=read(out/'v2_multicycle_audit.json');assert task['trajectory_sha256']==record['trajectory_sha256']
    acc=np.linalg.norm(z['solver_qacc'][:,:3],axis=1);later=t>first_end+1e-7
    report=dict(passed=all(checks.values()) and task['passed'],trace_checks_passed=all(checks.values()),task_passed=task['passed'],checks=checks,events=event_rows,
        expected_reference_activations=expected,event_identity='Original activation start time and hook; cycle independently reconstructed from the reference',
        cap_reconstruction_error=cap_error,vertical_force_alignment_error=vertical_error,qp_force_violation=force_violation,
        maximum_root_acceleration_after_first_cycle_m_s2=float(acc[later].max()) if np.any(later) else None,
        maximum_hook_force_after_first_cycle_n={g:float(np.linalg.norm(z['group_force_world_n'][later,j,:],axis=1).max()) for j,g in enumerate(record['groups']) if 'hook' in g} if np.any(later) else {},
        manifest_sha256=sha(out/'manifest.json'),trajectory_sha256=record['trajectory_sha256'],telemetry_sha256=record['telemetry_sha256'],trace_sha256=record['trace_sha256'],
        task_audit_sha256=sha(out/'v2_multicycle_audit.json'),source_sha256=sha(__file__),
        scope='Independent cap-curve, feedback-duration, QP-force-bound and telemetry consistency checks, plus the separate physical/policy audit. Controller caps constrain estimated forces, not physical collision peaks. This is a development candidate with ideal feedback, not an adopted controller or hardware safety certificate.')
    write(report_path,report);(out/(report_path.stem+'_source.py')).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
