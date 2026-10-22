"""Right-leg residual during weight transfer, including the left-foot unload.

Separate task from easy right-foot landing. Uses recorded qpos/qvel at 49 s and
requires three remaining supports confirmed, with the left foot unloaded.
"""
import numpy as np
from training.foot_residual_env import FootResidualEnv


class LoadTransferResidualEnv(FootResidualEnv):
    def __init__(self,suite,cases=None):
        super().__init__(suite,cases)
        self.start_time=49.;self.end_time=61.56
        self.max_steps=int(round((self.end_time-self.start_time)/.02))
        for name in self.cases:
            with np.load(self.suite/name/'trajectory.npz') as z:
                i=int(np.argmin(abs(z['time_s']-self.start_time)))
                if abs(z['time_s'][i]-self.start_time)>1e-6:raise ValueError('Recorded prefix must reach 49 s')
                self.prefixes[name]=(z['qpos'][i].copy(),z['qvel'][i].copy(),self.prefixes[name][2])

    def reset(self,*,seed=None,options=None):
        obs,info=super().reset(seed=seed,options=options)
        self.guard.expected=['rung_00','rung_01','rung_05','rung_05']
        self.left_force_tail=[]
        info.update(contract='load-transfer-v1-76obs-6act',
                    prefix='Recorded qpos/qvel at 49 s; solver warm start reset; residual enabled with 200 Hz low-pass filtering')
        return obs,info

    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        t=float(self.env.data.time)
        self.left_force_tail.append((t,self.env.contact_metrics()['left_foot_vertical_n']))
        self.left_force_tail=[row for row in self.left_force_tail if row[0]>=t-.25-1e-8]
        if truncated:
            history=self.guard.history
            confirmed=bool(history and t-history[0][0]>=.25-1e-7 and
                           all(all(valid[i] for i in (1,2,3)) for _,valid in history))
            unloaded=len(self.left_force_tail)>=12 and max(abs(f) for _,f in self.left_force_tail)<1.
            success=confirmed and unloaded
            # Base foot task requires four contacts; replace its terminal bonus.
            reward += (20. if success else -20.)-(20. if info['success'] else -20.)
            info.update(success=success,reason='success' if success else 'unconfirmed_loaded_tripod',
                        terminal_left_foot_peak_n=max(abs(f) for _,f in self.left_force_tail))
        info['scope']='49..61.56 s load transfer and left-foot unload; right-leg residual; finite-prefix training; ideal force feedback'
        return obs,reward,terminated,truncated,info
