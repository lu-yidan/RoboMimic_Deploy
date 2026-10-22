"""Verify instrumentation equality and distinguish targets from realized motion."""
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    root=Path('outputs/highrod_dynamics_trace_v1');p=root/'true_height_12.5mm'
    base=Path('outputs/highrod_height_mismatch_v1/true_height_12.5mm')
    with np.load(p/'trajectory.npz') as x,np.load(base/'trajectory.npz') as y:
        exact={k:bool(np.array_equal(x[k],y[k])) for k in ('qpos','qvel','time_s','actions','observations','torques','rewards')}
        assert all(exact.values()),exact
        velocity=y['qvel'];sample_times=y['time_s']
    with np.load(p/'dynamics.npz') as z:data={k:z[k].copy() for k in z.files}
    m=json.loads((p/'manifest.json').read_text());env=FullCycleResidualEnv(plan=m['reference']);e=env.env
    try:
        t=data['time_s'];dt=data['dt_s'];acc=data['average_acceleration'];desired=data['desired_acceleration']
        reconstructed=np.vstack([velocity[0,e.vadr],velocity[0,e.vadr]+np.cumsum(acc*dt[:,None],axis=0)])
        index=np.searchsorted(t,sample_times[1:]);index=np.minimum(index,len(t)-1)
        assert np.max(abs(t[index]-sample_times[1:]))<1e-10
        error=float(abs(reconstructed[index+1]-velocity[1:,e.vadr]).max())
        assert error<1e-9
        reference_velocity=np.array([env.reference(float(time-step))[1][e.vadr] for time,step in zip(t,dt)])
        target=100*data['reference_error']+20*(reference_velocity-reconstructed[:-1])
        target_error=float(abs(target-desired).max());assert target_error<1e-7
        k=[e.model.joint(j).name for j in e.joints].index('right_ankle_pitch_joint')
        keep=(t>=59)&(t<=61);opposed=keep&(desired[:,k]>0)&(acc[:,k]<0)
        rows=[]
        for requested in (59.,59.5,59.8,60.,60.1,float(t[-1])):
            i=int(np.argmin(abs(t-requested)))
            rows.append(dict(time_s=float(t[i]),target_ankle_acc_rad_s2=float(desired[i,k]),
                realized_average_ankle_acc_rad_s2=float(acc[i,k]),ankle_velocity_rad_s=float(reconstructed[i+1,k]),
                ankle_torque_nm=float(data['torque'][i,k]),ankle_constraint_generalized_force_nm=float(data['constraint_force'][i,k])))
        result=dict(passed=True,joint_name='right_ankle_pitch_joint',motor_index=k,all_trajectory_arrays_exact=exact,integrated_velocity_max_error=error,target_reconstruction_max_error=target_error,
            window_s=[59.,float(t[-1])],ankle_target_realized_rmse_rad_s2=float(np.sqrt(np.mean((desired[keep,k]-acc[keep,k])**2))),
            positive_target_negative_realized_duration_s=float(dt[opposed].sum()),samples=rows,
            dynamics_sha256=sha(p/'dynamics.npz'),trajectory_sha256=sha(p/'trajectory.npz'),baseline_trajectory_sha256=sha(base/'trajectory.npz'),source_sha256=sha(__file__),
            scope='Exact original trajectory with read-only instrumentation. Realized acceleration is velocity change per actual control interval; constraint force is MuJoCo generalized constraint force, not a measured foot wrench. Tracking mismatch alone does not identify its causal source.')
        (root/'diagnostic.json').write_text(json.dumps(result,indent=2));(root/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps(result,indent=2))
    finally:env.close()


if __name__=='__main__':main()
