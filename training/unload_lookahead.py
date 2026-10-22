"""Short model-based feasibility prediction in an independent MuJoCo instance."""
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_cycle import CycleTorqueController
from training.climb_robustness import apply_rung_offsets


class UnloadLookahead:
    def __init__(self, live, offsets, horizon=.1):
        if not np.isfinite(horizon) or not .005<=horizon<=.25 or abs(round(horizon/.005)*.005-horizon)>1e-9:
            raise ValueError('Lookahead must be 5..250 ms in 5 ms increments')
        self.env=WallSupportEnv(live.bundle);self.env.reset()
        apply_rung_offsets(self.env,offsets)
        self.horizon=float(horizon)
        self.spec=mujoco.mjtState.mjSTATE_INTEGRATION
        self.state=np.empty(mujoco.mj_stateSize(live.model,self.spec))
        if len(self.state)!=mujoco.mj_stateSize(self.env.model,self.spec):raise ValueError('Shadow state mismatch')
        self.controller=CycleTorqueController(self.env,torque_weight=10000.)
        self.calls=0

    def predict(self, live, reference, leg_kp):
        self.calls+=1;e=self.env
        mujoco.mj_getState(live.model,live.data,self.state,self.spec)
        mujoco.mj_setState(e.model,e.data,self.state,self.spec)
        self.controller.leg_kp=leg_kp
        start=float(e.data.time)
        for _ in range(round(self.horizon/.005)):
            q,v,active,phase=reference(float(e.data.time))
            # The experiment is explicitly scoped to an ongoing left unload.
            if phase!='lf_unload':break
            try:torque,_=self.controller.command(q,v,active)
            except RuntimeError as exc:
                return dict(failed=True,predicted_time_s=float(e.data.time),lead_s=float(e.data.time-start),reason=str(exc))
            e.data.ctrl[e.motors]=torque
            for _ in range(10):mujoco.mj_step(e.model,e.data)
            if not np.isfinite(e.data.qpos).all() or np.any(e.data.warning.number):
                return dict(failed=True,predicted_time_s=float(e.data.time),lead_s=float(e.data.time-start),reason='shadow_physics_failure')
        return dict(failed=False,predicted_time_s=float(e.data.time),lead_s=float(e.data.time-start))

    def close(self):self.env.close()
