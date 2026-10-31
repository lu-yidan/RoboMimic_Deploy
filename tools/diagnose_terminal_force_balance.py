"""Cold reconstructed-state diagnostic of QP vs MuJoCo acceleration.

No rollout is advanced. Omitted solver warm-start history and reconstructed
reference filtering prevent this from being an exact replay of live acceleration.
"""
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.closure_constrained_acceleration import solve_probe
from tools.continue_vector_full_cycle_training import sha


def main():
    source=Path('outputs/constrained_closure_screen_v1/gain0_drop1_foot_z_plus15')
    out=Path('outputs/terminal_qp_force_balance_20261023_v1');out.mkdir(exist_ok=False)
    audit=json.loads((source/'independent_audit.json').read_text())
    if not audit['integrity_passed'] or sha(source/'trajectory.npz')!=audit['trajectory_sha256']:raise ValueError('Audited trajectory required')
    b=FullCycleResidualEnv(plan='outputs/support_retarget_plan_v1/plan.npz',cases={'high':{'rung_01':[0.,0.,.015]}})
    b.reset(options={'case':'high'});e=b.env
    with np.load(source/'trajectory.npz') as f:z={k:f[k].copy() for k in f.files}
    indices=[int(np.argmin(abs(z['time_s']-t))) for t in (108.,110.,112.)]
    residual=np.zeros(26);saved={}
    for i in range(max(indices)+1):
        if i in indices:saved[i]=residual.copy()
        for _ in range(4):residual+=.05*(.015*z['actions'][i]-residual)
    rows=[]
    try:
        for i in indices:
            mujoco.mj_resetData(e.model,e.data)
            e.data.qpos[:]=z['qpos'][i];e.data.qvel[:]=z['qvel'][i];e.data.time=z['time_s'][i]
            mujoco.mj_forward(e.model,e.data)
            q,v,active,_=b.reference(float(e.data.time))
            old=saved[i];new=old+.05*(.015*z['actions'][i]-old)
            q[e.qadr[b.indices]]+=new;v[e.vadr[b.indices]]+=(new-old)/.005
            r=solve_probe(e,e.data,q,v,[b.contact.ik.bodies[k] for k,on in zip(('lf','rf','lh','rh'),active) if on])
            if not r['passed']:raise ValueError(r['status'])
            before=e.data.qpos.copy();before_time=float(e.data.time)
            e.data.ctrl[e.motors]=np.clip(r['torque'],e.limits[:,0],e.limits[:,1]);mujoco.mj_forward(e.model,e.data)
            actual=e.data.qacc.copy();aids=np.r_[2,e.vadr]
            mass=np.zeros((e.model.nv,e.model.nv));mujoco.mj_fullM(e.model,mass,e.data.qM)
            modeled_friction=np.zeros(e.model.nv)
            modeled_friction[e.vadr]=.3*np.tanh((v[e.vadr]+5*(q[e.qadr]-e.data.qpos[e.qadr]))/.02)
            applied=np.zeros(e.model.nv);applied[e.vadr]=r['torque']
            planned_constraint=mass@r['acceleration']+e.data.qfrc_bias+modeled_friction-applied
            components={
                'constraint_force_difference':e.data.qfrc_constraint-planned_constraint,
                'passive_force':e.data.qfrc_passive.copy(),
                'modeled_friction_correction':modeled_friction,
                'actuation_difference':e.data.qfrc_actuator-applied}
            responses={key:np.linalg.solve(mass,value) for key,value in components.items()}
            reconstructed=sum(responses.values())
            balance_error=float(abs(reconstructed-(actual-r['acceleration'])).max())
            physics_balance=float(abs(mass@actual+e.data.qfrc_bias-e.data.qfrc_actuator-e.data.qfrc_passive-e.data.qfrc_constraint).max())
            if balance_error>1e-7 or physics_balance>1e-7:raise ValueError('Force balance decomposition does not close')
            position=np.r_[e.data.qpos[2],e.data.qpos[e.qadr]];target=np.r_[e.initial[2]+.25,e.initial[e.qadr]]
            width=np.r_[.008,np.full(29,.038)]
            def violation(acc):return float(max(0.,np.max(abs(acc[aids]+4*e.data.qvel[aids]+4*(position-target))-4*width)))
            rows.append(dict(force_balance_error=physics_balance,acceleration_decomposition_error=balance_error,
                root_acceleration_difference_components={k:float(a[2]) for k,a in responses.items()},
                generalized_force_norms={k:float(np.linalg.norm(a)) for k,a in components.items()},
                time_s=before_time,root_rise_m=float(e.data.qpos[2]-e.initial[2]),
                predicted_root_acceleration_m_s2=float(r['acceleration'][2]),reconstructed_mujoco_root_acceleration_m_s2=float(actual[2]),
                model_barrier_max_violation=violation(r['acceleration']),reconstructed_barrier_max_violation=violation(actual),
                qp_constraint_residual=r['max_constraint_residual'],no_state_advance=bool(np.array_equal(before,e.data.qpos) and before_time==e.data.time)))
        result=dict(rows=rows,source_trajectory_sha256=sha(source/'trajectory.npz'),source_sha256=sha(__file__),
            scope='Cold reconstructed-state force-balance identity. Constraint forces include all MuJoCo constraints, not just contacts. Algebraic attribution is not causal proof or live replay; no rollout state was changed.')
        (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps(result,indent=2))
    finally:b.close()


if __name__=='__main__':main()
