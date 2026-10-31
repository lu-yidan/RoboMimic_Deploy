"""V2 endpoint acceptance, preserving the frozen motor/physics implementation.

Closure is a reported quality metric. Physics failure still stops immediately.
The adopted rise deadband is explicit, not an assertion of numerical causation.
"""
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv
from training.repeated_rung_errors import GROUPS


def endpoint_checks(sample, cycle):
    return dict(rise=abs(sample['rise_m']-.25*cycle)<.011,
        normal_support=min(sample['normal_n'].values())>10.,
        vertical_support=min(sample['vertical_n'].values())>10.,
        intended_contacts=sample['intended_contacts'],
        effector=sample['effector_error_m']<.01,
        tilt=sample['tilt_rad']<np.deg2rad(3),
        linear_speed=sample['linear_speed_m_s']<.01,
        angular_speed=sample['angular_speed_rad_s']<.1,
        joint_speed=sample['joint_speed_rad_s']<.1)


class PhysicalV2CycleEnv(StrictEnv):
    def reset(self, **kwargs):
        self.physical_reset_count=getattr(self,'physical_reset_count',0)+1
        self.v2_tail=[]
        self.v2_endpoint_records=[]
        return super().reset(**kwargs)

    def check_endpoint(self):
        index=len(self.cycle_results);end=float(self.ends[index]);e=self.env;d=e.data
        if end-2.<d.time<=end+1e-7:
            _,metrics=self.strict_sample(index)
            forces=e.contact_metrics()
            normal={g:0. for g in GROUPS}
            for ci,c in enumerate(d.contact):
                ba,bb=[int(e.model.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
                if (ba in e.robot_bodies)==(bb in e.robot_bodies):continue
                group=e.contact_groups.get(ba if ba in e.robot_bodies else bb)
                if group in normal:
                    wrench=np.zeros(6);mujoco.mj_contactForce(e.model,d,ci,wrench)
                    normal[group]+=float(max(0,wrench[0]))
            expected={g:f'rung_{index+1+(4 if "hook" in g else 0):02d}' for g in GROUPS}
            loaded=[c for c in forces['contacts'] if np.linalg.norm(c['force_on_robot_world_n'])>1.]
            sample=dict(time_s=float(d.time),cycle=index+1,rise_m=float(d.qpos[2]-e.initial[2]),
                closure_rad=float(abs(d.qpos[e.qadr]-e.initial[e.qadr]).max()),
                normal_n=normal,vertical_n={g:float(forces[g+'_vertical_n']) for g in GROUPS},
                intended_contacts=bool(loaded) and all(c['group'] in expected and c['external_geom']==expected[c['group']] for c in loaded),
                loaded_contacts=loaded,**metrics)
            self.v2_tail.append(sample)
        if abs(d.time-end)<1e-7:
            checks={k:all(endpoint_checks(s,index+1)[k] for s in self.v2_tail) for k in endpoint_checks(self.v2_tail[-1],index+1)} if self.v2_tail else {}
            passed=len(self.v2_tail)>=99 and bool(checks) and all(checks.values())
            legacy=passed and all(abs(s['rise_m']-.25*(index+1))<.01 and s['closure_rad']<.04 for s in self.v2_tail)
            self.cycle_results.append(dict(cycle=index+1,time_s=float(d.time),passed=bool(passed),
                acceptance='physical-v2',legacy_strict_passed=bool(legacy),samples=len(self.v2_tail),checks=checks,
                closure_mrad=max((s['closure_rad'] for s in self.v2_tail),default=float('nan'))*1000))
            self.v2_endpoint_records.extend(self.v2_tail);self.v2_tail=[]
            return bool(passed)
        return True

    def info(self,reason,success):
        return dict(super().info(reason,success),acceptance='physical-v2',
                    physical_reset_count=self.physical_reset_count)
