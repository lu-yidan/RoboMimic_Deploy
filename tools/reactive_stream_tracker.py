"""Consume incrementally released GMR poses; advance one physical G1 in virtual time.

No DDS. No full future opponent/reference file is loaded. Missing preview holds the
last available generated pose. Waiting for generation pauses virtual simulation.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand, get_gravity_orientation
from FSM.FSM import FSM
from deploy_mujoco.deploy_mujoco import pd_control
from prepare_duobox_scalebfm import convert


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', type=Path, required=True)
    a = p.parse_args(); job = a.job.resolve()
    request = json.loads((job/'request.json').read_text())
    torch.set_num_threads(1)
    gmr = ROOT.parent/'GMR'; scale = ROOT.parent/'ScaleBFM'
    from common.deployment_config import load_scalebfm_config
    cfg = OmegaConf.load(ROOT/'deploy_mujoco/config/mujoco.yaml')
    cfg.scalebfm = load_scalebfm_config(require_enabled=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT/'g1_description/scene.xml'))
    model.opt.timestep = .002
    data = mujoco.MjData(model)
    state, output = StateAndCmd(29), PolicyOutput(29)
    names = [model.joint(int(model.actuator_trnid[i,0])).name for i in range(29)]
    gmodel = mujoco.MjModel.from_xml_path(str(gmr/'assets/unitree_g1/g1_mocap_29dof.xml'))
    gindices = [gmodel.joint(n).qposadr[0] for n in names]
    actual, timing = [], []
    fsm = None; processed = 0; last_end = 0; failure = None
    started = time.monotonic(); wait_start = started
    while True:
        chunks = sorted(job.glob('robot_*.npz'))
        chunks = [x for x in chunks if int(x.stem.split('_')[1]) > last_end]
        if not chunks:
            if (job/'producer.done').exists(): break
            if (job/'abort').exists(): raise RuntimeError('Pipeline aborted')
            if time.monotonic()-wait_start > 600: raise TimeoutError('No retarget chunk')
            time.sleep(.05); continue
        chunk = chunks[0]; end = int(chunk.stem.split('_')[1])
        pair = np.load(chunk); actor = str(pair['actors'][0])
        reference = job/('reference_%04d.npz' % end)
        convert(chunk, actor, reference, gmr, scale)
        archive = dict(np.load(reference))
        if fsm is None:
            cfg.scalebfm.motion = str(reference)
            fsm = FSM(state, output, cfg.scalebfm, names, enable_freekick=False)
            tracker = fsm.scalebfm_policy
            # One initial-state reset; no support forces or resets during tracking.
            data.qpos[:7] = pair['qpos'][0,0,:7]
            data.qpos[7:36] = pair['qpos'][0,0,gindices]
            data.qvel[:] = 0
            mujoco.mj_forward(model,data)
            fsm.cur_policy = tracker
            state.q = data.qpos[7:36].copy(); state.dq = data.qvel[6:35].copy()
            state.pelvis_pos_w = data.qpos[:3].astype(np.float32)
            state.pelvis_quat_w = data.qpos[3:7].astype(np.float32)
            tracker.enter()
        else:
            # Update reference buffer only: preserve action history, episode time,
            # and physical state. World frame stays fixed across all chunks.
            tracker.env._setup_motion(archive)
            tracker.motion_len = len(archive['joint_pos'])
        segment_start = time.perf_counter()
        for t in range(processed, len(archive['joint_pos'])):
            state.q = data.qpos[7:36].copy(); state.dq = data.qvel[6:35].copy()
            state.pelvis_pos_w = data.qpos[:3].astype(np.float32)
            state.pelvis_quat_w = data.qpos[3:7].astype(np.float32)
            state.root_ang_vel_b = data.qvel[3:6].astype(np.float32)
            state.gravity_ori = get_gravity_orientation(data.qpos[3:7])
            state.skill_cmd = FSMCommand.INVALID
            tracker.run()
            q = np.zeros(gmodel.nq); q[:7] = data.qpos[:7]; q[gindices] = data.qpos[7:36]
            actual.append(q)
            for _ in range(10):
                data.ctrl[:] = np.clip(pd_control(output.actions,data.qpos[7:36],output.kps,
                                      np.zeros(29),data.qvel[6:35],output.kds),-np.array(cfg.tau_limit),cfg.tau_limit)
                mujoco.mj_step(model,data)
            mujoco.mj_forward(model,data)
            tilt = np.degrees(np.arccos(np.clip(data.body('pelvis').xmat[8],-1,1)))
            if tracker.failed or data.qpos[2] < .3 or tilt > 70 or not np.isfinite(q).all():
                failure = 'fall_or_policy_failure'; break
        processed = len(actual)
        timing.append(dict(end=end, observed_until=int(pair['observed_until']),
                           generation_ms=float(pair['prediction_ms']), retarget_ms=float(pair['retarget_ms']),
                           tracking_ms=(time.perf_counter()-segment_start)*1000))
        # Browser uses a shared 30 Hz timeline; interpolate physical 50 Hz states.
        count = min(end, int((len(actual)-1)*30/50)+1)
        times = np.arange(count)/30; actual_np = np.array(actual)
        q_actual = np.column_stack([np.interp(times,np.arange(len(actual))/50,actual_np[:,j]) for j in range(gmodel.nq)])
        q_actual[:,3:7] /= np.linalg.norm(q_actual[:,3:7],axis=1,keepdims=True)
        qpair = np.stack([q_actual,pair['qpos'][:count,1]],1)
        temp = job/'pair.tmp.npz'
        np.savez_compressed(temp,qpos=qpair,fps=30,actors=np.array([actor+' · 生成+物理跟踪',str(pair['actors'][1])+' · 预录运动学']))
        temp.replace(job/'pair_motion.npz')
        meta = dict(actors=[actor+' · 生成+物理跟踪',str(pair['actors'][1])+' · 预录运动学'],
                    fps=30,frames=count,seconds=count/30,dynamics=True,diagnostics={},
                    method='causal Ready-to-React → streaming GMR → ScaleBFM → MuJoCo',
                    online_realtime=False,upper_actual_state_feedback=False,opponent_physics=False,
                    future_preview='hold latest available generated frame',warmup_frames=4,
                    initialization='reference pose once at startup; no external support',
                    timing=timing,failure=failure,wall_seconds=time.monotonic()-started,
                    note='逐块因果生成；等待计算时暂停虚拟时间；非实时闭环。蓝色物理仿真，橙色运动学；无双方碰撞。')
        temp = job/'metadata.tmp.json'; temp.write_text(json.dumps(meta)); temp.replace(job/'metadata.json')
        (job/('ack_%04d' % end)).touch(); last_end = end; wait_start = time.monotonic()
        print('TRACKED',end,'physical frames',processed,flush=True)
        if failure:
            (job/'abort').write_text(failure); break
    np.savez_compressed(job/'physical_50hz.npz',qpos=np.array(actual),fps=50)
    (job/'tracker.done').touch()

if __name__ == '__main__': main()
