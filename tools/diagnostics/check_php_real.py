"""Offline PHP + real bridge runtime validation. No camera, DDS or motor transport.

Uses actual ONNX models and BridgeCmd serialization; synthetic robot state is NOT
an observation of a physical robot. Optional raw depth recordings keep their age.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
from deploy_real.config import Config
from common.deployment_config import load_php_config, load_php_real_config
from common.remote_controller import KeyMap
from common.utils import FSMStateName
from policy.php_parkour.real_depth import load_camera_config, DepthBuffer, Intrinsics
from bridge.python.policy_runtime import PolicyRuntime
from bridge.python.bridge_state_dds import BridgeState
from bridge.python.bridge_cmd_dds import BridgeCmd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-config',type=Path,help='Defaults to the selected deploy config camera_config')
    parser.add_argument('--recording',type=Path,help='NPZ from preview_php_depth.py (raw optical-Z metres + intrinsics)')
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--out',type=Path,default=Path('/tmp/php-real-offline.json'))
    args=parser.parse_args()
    if not 1 <= args.steps <= 10000:
        parser.error('steps must be in [1,10000]')
    cfg=Config();cfg.log_enabled=False;cfg.scalebfm.enabled=False
    cfg.php=load_php_config();cfg.php['enabled']=True
    camera=load_camera_config(args.camera_config or load_php_real_config()['camera_config']);cfg.php['camera']=camera
    if args.recording:
        with np.load(args.recording,allow_pickle=False) as archive:
            depth=archive['depth_m'].copy()
            values=archive['intrinsics']
            k=Intrinsics(int(values[0]),int(values[1]),*map(float,values[2:]))
            age=float(archive['capture_age_s'])
            if not np.allclose(archive['position_m'],camera['mount']['position_m']) or not np.allclose(archive['rpy_deg'],camera['mount']['rpy_deg']):
                raise ValueError('Recording mount does not match camera config')
    else:
        f=53/np.tan(np.deg2rad(89.5/2))
        k=Intrinsics(106,60,f,f,53,30)
        depth=np.full((60,106),1.5,np.float32);age=0.
    source=DepthBuffer(camera)
    canonical=yaml.safe_load((ROOT/'policy/php_parkour/config/php.yaml').read_text())
    q=[canonical['default_joint_pos'][canonical['joint_names'].index(n)] for n in cfg.joint_names]
    step_id=0
    times=[]
    # Even accidental future transport creation must fail in this tool.
    with patch('cyclonedds.domain.DomainParticipant.__init__',side_effect=AssertionError('Offline tool cannot open DDS')):
        runtime=PolicyRuntime(cfg,connect=False,depth_source=source)
        try:
            def step(keys=(),fresh=True):
                nonlocal step_id
                step_id+=1
                if fresh:
                    source.submit(depth,k,time.monotonic()-age,step_id)
                raw=bytearray(24);struct.pack_into('<H',raw,2,sum(1<<key for key in keys))
                state=BridgeState(tick=step_id,q=q,dq=[0.]*29,imu_quat_wxyz=[1.,0,0,0],
                                  imu_gyro=[0.]*3,remote_raw=list(raw))
                begin=time.monotonic();frame=runtime.step(state);times.append((time.monotonic()-begin)*1000)
                return frame
            step()
            frame=step([KeyMap.L1,KeyMap.A])
            if runtime.FSM_controller.cur_policy.name != FSMStateName.SKILL_PHP_PARKOUR:
                raise RuntimeError('PHP entry rejected: '+source.latest().error)
            for _ in range(args.steps):
                frame=step([KeyMap.L1,KeyMap.A])
                assert not frame.request_damping and np.isfinite(frame.q_des).all()
                torque=frame.kp*(frame.q_des-runtime.state_cmd.q)-frame.kd*runtime.state_cmd.dq
                assert np.all(np.abs(torque)<=runtime._php_effort_limits+1e-4)
                decoded=BridgeCmd.deserialize(BridgeCmd(seq=frame.seq,q_des=frame.q_des.tolist(),
                    kp=frame.kp.tolist(),kd=frame.kd.tolist()).serialize())
                np.testing.assert_allclose(decoded.q_des,frame.q_des)
            source.invalidate('offline injected camera disconnect')
            frame=step([KeyMap.L1,KeyMap.A],fresh=False)
            assert frame.request_damping and not frame.kp.any()
            step([KeyMap.L1,KeyMap.A])
            assert runtime.FSM_controller.cur_policy.name==FSMStateName.PASSIVE
            step();step([KeyMap.L1,KeyMap.A])
            assert runtime.FSM_controller.cur_policy.name==FSMStateName.SKILL_PHP_PARKOUR
            step([KeyMap.Y])
            assert runtime.FSM_controller.cur_policy.name==FSMStateName.SKILL_SMP_RECOVERY
            report=dict(offline=True,dds_participants=0,motor_commands_sent=0,hardware_capture_verified=False,
                input='recorded depth' if args.recording else 'synthetic plane depth',
                camera_config=camera,policy_steps=args.steps,wire_roundtrip=True,pd_effort_bounds=True,
                loss_damps_same_tick=True,no_auto_reentry=True,y_preempts_php=True,
                runtime_ms_p50=float(np.percentile(times,50)),runtime_ms_p95=float(np.percentile(times,95)),
                scope='Static synthetic robot state; no physical robot or camera validation')
            args.out.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2))
        finally:
            runtime.close()


if __name__=='__main__':
    main()
