"""No-DDS startup and static-state CUDA timing check for the real deployment stack."""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from deploy_real.config import Config
from common.ctrlcomp import StateAndCmd, PolicyOutput
from FSM.FSM import FSM
from policy.scalebfm.motion_catalog import MotionCatalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1: parser.error('--steps must be positive')
    # Importing the entry checks bridge imports, but never constructs runtime or DDS channels.
    importlib.import_module('bridge.python.deploy_policy')
    cfg = Config()
    torch.set_num_threads(cfg.torch_threads)
    state, output = StateAndCmd(29), PolicyOutput(29)
    fsm = FSM(state, output, cfg.scalebfm, cfg.joint_names, enable_freekick=False)
    tracker = fsm.scalebfm_policy
    if tracker is None: raise ValueError('ScaleBFM must be enabled for this check')
    catalog = MotionCatalog(tracker.motion_path, cfg.scalebfm.motion_globs, Path(__file__).resolve().parents[2])
    tracker.preload_motions(catalog.paths)
    state.q = fsm.fixed_pose_1._default_out.copy()
    state.pelvis_quat_w = np.array([1., 0., 0., 0.], dtype=np.float32)
    entry_ms, step_ms = [], []
    for path in catalog.paths:
        tracker.load_motion(path)
        torch.cuda.synchronize()
        start = time.perf_counter()
        tracker.enter()
        torch.cuda.synchronize()
        entry_ms.append((time.perf_counter()-start)*1000)
        if tracker.agent.config.get('cuda_graph', False):
            with torch.inference_mode():
                # Compare changed references as well as the initial captured inputs.
                graph_output = tuple(v.clone() for v in tracker.agent.get_action(tracker.obs))
                tracker.agent.config.cuda_graph = False
                try:
                    native_output = tracker.agent.get_action(tracker.obs)
                finally:
                    tracker.agent.config.cuda_graph = True
                for actual, expected in zip(graph_output, native_output):
                    if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-4):
                        raise RuntimeError('CUDA graph parity failed for '+str(path))
        for _ in range(args.steps):
            start = time.perf_counter()
            tracker.run()
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter()-start)*1000)
            if tracker.failed or not np.isfinite(output.actions).all():
                raise RuntimeError('Inference failed for '+str(path))
            # References advance each step: verify graph inputs are refreshed,
            # not accidentally frozen at the captured first observation.
            if len(step_ms) in (1, 10, 50) and tracker.agent.config.get('cuda_graph', False):
                with torch.inference_mode():
                    graph_output = tuple(v.clone() for v in tracker.agent.get_action(tracker.obs))
                    tracker.agent.config.cuda_graph = False
                    try:
                        native_output = tracker.agent.get_action(tracker.obs)
                    finally:
                        tracker.agent.config.cuda_graph = True
                    for actual, expected in zip(graph_output, native_output):
                        if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-4):
                            raise RuntimeError('Advancing-reference CUDA graph parity failed')
        tracker.exit()
    print(json.dumps(dict(no_dds=True, motions=len(catalog.paths), samples=len(step_ms),
                         torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                         step_ms_p50=float(np.percentile(step_ms,50)),
                         step_ms_p95=float(np.percentile(step_ms,95)),
                         step_ms_max=max(step_ms), entry_ms_max=max(entry_ms),
                         budget_ms=cfg.control_dt*1000,
                         over_budget=sum(v>cfg.control_dt*1000 for v in step_ms),
                         scope='static state inference; excludes DDS, motor IO and closed-loop tracking'),indent=2))

if __name__ == '__main__': main()
