"""Report whole-cycle support impulses and COM momentum balance.

The result is a diagnostic of saved simulation telemetry. It is not a hardware
load rating, a new success threshold, or a control/learning intervention.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from tools.validate_swing_reference import read,write,sha


def analyze(run,report_name):
    audit=read(run/report_name);m=read(run/'manifest.json');summary=read(run/'summary.json')
    assert audit['passed'] and audit['trajectory_sha256']==sha(run/'trajectory.npz')
    assert audit['telemetry_sha256']==sha(run/'physics_telemetry.npz')
    groups=audit.get('groups') or read(run/'hook_admission_record.json')['groups']
    with np.load(run/'trajectory.npz') as f:poses={k:f[k] for k in ('qpos','qvel','time_s')}
    with np.load(run/'physics_telemetry.npz') as f:
        data={k:f[k] for k in ('time_s','group_force_world_n','solver_qacc','motor_torque_nm')}
    with np.load(run/'reference.npz') as f:ends=f['cycle_ends_s']
    e=WallSupportEnv(Path(m['bundle']));mass=float(e.model.body_subtreemass[e.pelvis]);gravity=e.model.opt.gravity.copy()
    assert abs(mass*np.linalg.norm(gravity)-e.robot_weight_n)<1e-9
    dt=float(e.model.opt.timestep);assert dt==.0005
    assert abs(data['time_s'][-1]-ends[-1])<1e-7
    def com_velocity(t):
        index=int(np.argmin(abs(poses['time_s']-t)))
        assert abs(poses['time_s'][index]-t)<1e-7
        e.data.qpos[:]=poses['qpos'][index];e.data.qvel[:]=poses['qvel'][index]
        mujoco.mj_forward(e.model,e.data)
        jac=np.zeros((3,e.model.nv));mujoco.mj_jacSubtreeCom(e.model,e.data,jac,e.pelvis)
        return jac@e.data.qvel
    rows=[]
    try:
        for i,end in enumerate(ends):
            start=float(ends[i-1]) if i else 0.
            mask=(data['time_s']>start+1e-7)&(data['time_s']<=end+1e-7)
            count=int(mask.sum());duration=count*dt
            assert abs(duration-(end-start))<1e-8
            impulses=data['group_force_world_n'][mask].sum(axis=0)*dt
            gravity_impulse=mass*gravity*duration
            momentum_change=mass*(com_velocity(float(end))-com_velocity(start))
            residual=impulses.sum(axis=0)+gravity_impulse-momentum_change
            root_acc=data['solver_qacc'][mask,:3]
            torque=data['motor_torque_nm'][mask]
            rows.append(dict(cycle=i+1,duration_s=duration,samples=count,
                support_impulse_world_ns={g:impulses[j].tolist() for j,g in enumerate(groups)},
                average_vertical_support_n={g:float(impulses[j,2]/duration) for j,g in enumerate(groups)},
                net_external_impulse_ns=(impulses.sum(axis=0)+gravity_impulse).tolist(),
                com_momentum_change_kg_m_s=momentum_change.tolist(),
                impulse_minus_momentum_ns=residual.tolist(),
                residual_norm_ns=float(np.linalg.norm(residual)),
                root_acceleration_rms_m_s2=float(np.sqrt(np.mean(np.sum(root_acc**2,axis=1)))),
                joint_torque_rms_nm=np.sqrt(np.mean(torque**2,axis=0)).tolist(),
                summed_squared_torque_nm2_s=float(np.sum(torque**2)*dt),
                endpoint_closure_mrad=summary['cycle_results'][i]['closure_mrad']))
    finally:e.close()
    return dict(run=str(run),audit_name=report_name,audit_sha256=sha(run/report_name),
        trajectory_sha256=sha(run/'trajectory.npz'),telemetry_sha256=sha(run/'physics_telemetry.npz'),
        case=m['case'],plan_sha256=m['plan_sha256'],checkpoint_sha256=m['checkpoint_sha256'],
        mass_kg=mass,groups=groups,cycles=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    baseline=analyze(a.baseline.resolve(),'telemetry_audit.json')
    candidate=analyze(a.candidate.resolve(),'hook_admission_audit_v2.json')
    assert all(baseline[k]==candidate[k] for k in ('case','plan_sha256','checkpoint_sha256','mass_kg','groups'))
    assert len(baseline['cycles'])==len(candidate['cycles'])
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(baseline=baseline,candidate=candidate,source_sha256=sha(__file__),
        scope='Integrate signed world contact forces over every 0.5 ms solver interval. Reconstruct robot COM momentum at saved cycle boundaries using the subtree COM Jacobian, not floating-base velocity. Residuals are reported without a fitted acceptance threshold. Root acceleration RMS is a simulation metric; squared torque is only a control-effort proxy, not electrical energy or measured heat. No new training or automatic adoption.')
    write(a.output/'support_impulse_comparison.json',report)
    (a.output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({label:[{k:v for k,v in row.items() if k in ('cycle','average_vertical_support_n','residual_norm_ns','root_acceleration_rms_m_s2','summed_squared_torque_nm2_s')} for row in arm['cycles']]
                      for label,arm in [('baseline',baseline),('candidate',candidate)]}),flush=True)


if __name__=='__main__':main()
