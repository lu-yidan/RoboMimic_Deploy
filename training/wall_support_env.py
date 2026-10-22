"""Contact diagnostics on the rung wall, reusing the existing 100D/29D environment."""
from pathlib import Path
import mujoco
import numpy as np
from training.climb_env import ClimbSupportEnv

DEFAULT_WALL_BUNDLE=Path(__file__).resolve().parents[1]/'outputs/wall_support_bundle_v2'


class WallSupportEnv(ClimbSupportEnv):
    def __init__(self,bundle=DEFAULT_WALL_BUNDLE,**kwargs):
        super().__init__(bundle,**kwargs)
        self.contact_groups={self.model.body(name).id:label for name,label in
                             [('hook','left_hook'),('right_hook','right_hook'),
                              ('left_ankle_roll_link','left_foot'),('right_ankle_roll_link','right_foot')]}
        self.robot_bodies=set()
        for bid in range(1,self.model.nbody):
            parent=bid
            while parent and parent!=self.pelvis:parent=int(self.model.body_parentid[parent])
            if parent==self.pelvis:self.robot_bodies.add(bid)
        self.robot_weight_n=float(sum(self.model.body_mass[list(self.robot_bodies)])*np.linalg.norm(self.model.opt.gravity))

    def contact_metrics(self):
        # These are the last physics solver's contact forces, sampled at 50 Hz.
        # Normal points from geom1 to geom2; contact wrench acts on geom2.
        sums={name:np.zeros(3) for name in (*self.contact_groups.values(),'other_robot')}
        contacts=[]
        for index,c in enumerate(self.data.contact):
            a,b=int(c.geom1),int(c.geom2)
            ba,bb=int(self.model.geom_bodyid[a]),int(self.model.geom_bodyid[b])
            ina,inb=ba in self.robot_bodies,bb in self.robot_bodies
            if ina==inb:continue
            robot_geom,external,body,sign=(b,a,bb,1.) if inb else (a,b,ba,-1.)
            wrench=np.zeros(6)
            mujoco.mj_contactForce(self.model,self.data,index,wrench)
            force=sign*c.frame.reshape(3,3).T@wrench[:3]
            group=self.contact_groups.get(body,'other_robot');sums[group]+=force
            contacts.append(dict(robot_geom=self.model.geom(robot_geom).name,
                                 external_geom=self.model.geom(external).name,group=group,
                                 force_on_robot_world_n=force.tolist(),distance_m=float(c.dist)))
        result={name+'_vertical_n':float(value[2]) for name,value in sums.items()}
        result.update(robot_weight_n=self.robot_weight_n,
                      total_contact_vertical_n=float(sum(v[2] for v in sums.values())),
                      contacts=contacts,force_sample='Last physics solve, sampled at policy frequency; not a peak-load measurement')
        return result

    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        info.update(self.contact_metrics())
        return obs,reward,terminated,truncated,info
