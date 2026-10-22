"""Move the sole contact target during airborne swing, then retain it in stance."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.plan_clearance_cycle import audit_plan
from training.temporal_base_reference_env import envelope_and_rate
from tools.continue_vector_full_cycle_training import sha


def main():
    out=Path('outputs/landing_contact_plan_v3');out.mkdir(exist_ok=False)
    parent=Path('outputs/support_retarget_plan_v1/plan.npz');terminal=Path('outputs/terminal_contact_minimax_v1/height10.npz')
    env=FullCycleResidualEnv(plan=parent);e=env.env;m=e.model;d=mujoco.MjData(m)
    plan={k:v.copy() for k,v in env.plan.items()};original=plan['qpos'].copy();records=[]
    with np.load(terminal) as z:target=z['qpos'].copy();local=z['local_contact_x'].copy()
    try:
        for i,t in enumerate(plan['time']):
            if t>99.66:continue
            for side,indices,body,x,start,end in [('left',slice(0,6),env.endpoint_bodies[2],local[0],70.2,80.66),('right',slice(6,12),env.endpoint_bodies[3],local[1],37.,44.2)]:
                u=float(np.clip((t-start)/(end-start),0,1));gain=u*u*(3-2*u)
                if gain==0:continue
                qa=e.qadr[indices];js=e.joints[indices];q=plan['qpos'][i].copy();d.qpos[:]=q;mujoco.mj_kinematics(m,d)
                R=d.xmat[body].reshape(3,3).copy();position=d.xpos[body]+R@np.array([.09,0.,-.0498]);contact_x=.09-(.09-x)*gain;normal_y=R[1,2];yaw=np.arctan2(R[1,0],R[0,0])
                position[2]+=.01*gain-(.01*envelope_and_rate(float(t))[0] if side=="right" else 0.)
                def residual(values):
                    d.qpos[:]=q;d.qpos[qa]=values;mujoco.mj_kinematics(m,d)
                    Rnew=d.xmat[body].reshape(3,3);angle=np.arctan2(Rnew[1,0],Rnew[0,0])-yaw
                    return np.r_[d.xpos[body]+Rnew@np.array([contact_x,0.,-.0498])-position,.2*(Rnew[1,2]-normal_y),.2*np.arctan2(np.sin(angle),np.cos(angle)),1e-6*(values-q[qa])]
                lo=np.maximum(m.jnt_range[js,0]+1e-6,q[qa]-.2);hi=np.minimum(m.jnt_range[js,1]-1e-6,q[qa]+.2)
                fit=least_squares(residual,np.clip(q[qa],lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=200,gtol=1e-11,xtol=1e-11,ftol=1e-11)
                r=residual(fit.x);error=float(np.linalg.norm(r[:3]));rotation=float(np.linalg.norm(r[3:5])/.2)
                records.append(dict(time_s=float(t),side=side,position_error_m=error,rotation_error_rad=rotation))
                if error>1e-5 or rotation>1e-4:
                    (out/'failed_node.json').write_text(json.dumps(records[-1],indent=2));raise RuntimeError('Landing retarget rejected')
                plan['qpos'][i]=d.qpos.copy()
        start_index=int(np.argmin(abs(plan['time']-99.66)));assert abs(plan['time'][start_index]-99.66)<1e-8
        delta=plan['qpos'][start_index]-original[start_index]
        for i,t in enumerate(plan['time']):
            if t<=99.66:continue
            u=float(np.clip((t-99.66)/6.,0,1));w=u*u*(3-2*u)
            plan['qpos'][i]=original[i]+(1-w)*delta+w*(target-original[-1])
        np.testing.assert_array_equal(plan['qpos'][plan['time']<=37],original[plan['time']<=37])
        np.savez_compressed(out/'plan.npz',**plan)
        audits={}
        for name,height in [('nominal',0.),('estimated',.01)]:
            env.cases={'probe':{'rung_01':[0.,0.,height]}};env.reset(options={'case':'probe'})
            audits[name]=audit_plan(e,plan,dt=.005);audits[name]['inherited_parent_ik_residual']=audits[name].pop('ik_residual')
            (out/(name+'_audit.json')).write_text(json.dumps(audits[name],indent=2))
        manifest=dict(parent=str(parent),parent_sha256=sha(parent),terminal=str(terminal),terminal_sha256=sha(terminal),plan_sha256=sha(out/'plan.npz'),source_sha256=sha(__file__),local_contact_x_m=local.tolist(),node_checks=records,passed=all(a['passed'] for a in audits.values()),scope='New reference family: move contact location during swing and preserve through stance, then close to the minimax target. Same duration and physics acceptance. Geometry checks do not prove execution. No policy training.')
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'builder_source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps({name:dict(passed=a['passed'],failures=len(a['failures']),peak_reference_joint_speed_rad_s=a['peak_reference_joint_speed_rad_s']) for name,a in audits.items()}))
    finally:env.close()


if __name__=='__main__':main()
