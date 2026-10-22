"""Create isolated training sources with explicit actual-contact assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_new(name, text):
    target = ROOT / name
    if target.exists():
        raise FileExistsError(target)
    target.write_text(text, encoding='utf-8')


s = (ROOT / 'training/foot_residual_env.py').read_text(encoding='utf-8')
s = s.replace('from training.climb_cycle import CycleTorqueController', 'from training.contact_frame_controller import ContactFrameController')
s = s.replace('self.controller=CycleTorqueController(e,torque_weight=10000.)', 'self.controller=ContactFrameController(e,torque_weight=10000.)')
s = s.replace('self.last_torque=np.zeros(29)', '''self.last_torque=np.zeros(29)
        self.joint_minimum=float(np.min(np.minimum(e.data.qpos[e.qadr]-e.model.jnt_range[e.joints,0],e.model.jnt_range[e.joints,1]-e.data.qpos[e.qadr])))
        if not np.all(e.model.jnt_limited[e.joints]):raise ValueError('Expected bounded joints')
        if self.joint_minimum < -1e-7:raise ValueError('Recorded initial state violates native joint bounds')''')
s = s.replace('mujoco.mj_step(m,d);mujoco.mj_kinematics(m,d)', '''mujoco.mj_step(m,d);mujoco.mj_kinematics(m,d)
                margin=float(np.min(np.minimum(d.qpos[e.qadr]-m.jnt_range[e.joints,0],m.jnt_range[e.joints,1]-d.qpos[e.qadr])))
                self.joint_minimum=min(self.joint_minimum,margin)
                if margin < -1e-7:reason='native_joint_position_violation';break''')
s = s.replace("info=dict(success=success,reason=reason", "info=dict(minimum_native_joint_margin_rad=self.joint_minimum,controller_assumption='Exact MuJoCo contact points, normals, topology and ideal forces',success=success,reason=reason")
s = s.replace("contract='foot-residual-v1-76obs-6act'", "contract='contact-foot-residual-v1-76obs-6act'")
write_new('training/contact_foot_residual_env.py', s)
s = (ROOT / 'training/load_transfer_residual_env.py').read_text(encoding='utf-8')
s = s.replace('from training.foot_residual_env import FootResidualEnv', 'from training.contact_foot_residual_env import FootResidualEnv')
s = s.replace('load-transfer-v1-76obs-6act', 'contact-load-transfer-v1-76obs-6act')
s = s.replace('ideal force feedback', 'ideal force feedback and exact contact geometry for the low-level controller')
write_new('training/contact_load_transfer_residual_env.py', s)
s = (ROOT / 'tools/train_load_transfer.py').read_text(encoding='utf-8')
s = s.replace('training.load_transfer_residual_env', 'training.contact_load_transfer_residual_env')
s = s.replace("'training/load_transfer_residual_env.py','training/foot_residual_env.py'", "'training/contact_load_transfer_residual_env.py','training/contact_foot_residual_env.py','training/contact_frame_controller.py'")
s = s.replace("'tools/train_load_transfer.py'", "'tools/train_contact_load_transfer.py'")
s = s.replace("contract='76 observations", "controller='Exact MuJoCo contact geometry and topology; ideal contact forces. Native joint position ranges checked at every 0.5 ms physics step.',\n                  contract='76 observations")
write_new('tools/train_contact_load_transfer.py', s)
