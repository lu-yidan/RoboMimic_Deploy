"""Bounded ScaleBFM MuJoCo validation. Never imports DDS or opens robot transport."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalebfm-root", type=Path, default=ROOT.parent / "ScaleBFM")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--backend", choices=["torch", "tensorrt"], default="torch")
    parser.add_argument("--robot-xml", type=Path,
                        help="Omit for official ScaleBFM robot; use g1_description/scene.xml for deployment asset.")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if not 0 < args.seconds <= 120:
        parser.error("--seconds must be in (0,120]")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    bridge = args.scalebfm_root.resolve() / "ScaleBridge"
    checkpoint = (args.checkpoint or args.scalebfm_root / "artifacts/scalebfm/humanoid_transformer_m" / ("model_22200.pt" if args.backend == "torch" else "model_22200_tensorrt.pt")).resolve()
    motion = (args.motion or bridge / "scalebridge/data/motion/g1_29dof/squat.npz").resolve()
    robot = args.robot_xml.resolve() if args.robot_xml else None
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path[:0] = [str(bridge), str(bridge / "scalebridge")]
    os.chdir(bridge)

    import mujoco
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import open_dict

    with initialize_config_dir(config_dir=str(bridge / "scalebridge/config"), version_base=None):
        cfg = compose(config_name="base")
    with open_dict(cfg):
        cfg.agent.config.checkpoint = str(checkpoint)
        cfg.agent.config.backend = args.backend
        if args.backend == "torch":
            cfg.agent.config.metadata = str(checkpoint.parent / "model_22200_tensorrt_metadata.json")
            cfg.agent.config.mode_table = str(checkpoint.parent / "mode_table.pt")
            cfg.agent.config.xml_path = str(bridge / "scalebridge/data/robot/g1_29dof/g1_29dof.xml")
        cfg.env.config.motion_path = str(motion)
        cfg.simulator.config.headless = True
        cfg.simulator.config.record_video = args.video
        cfg.simulator.config.output_dir = str(out)
        if robot:
            cfg.asset.xml_path = str(robot)
    # This entry point intentionally has no backend override: simulation only.
    agent = instantiate(cfg.agent)
    env = instantiate(cfg.env, metadata_dict=agent.get_meta_data())
    sim = env.simulator
    effort_limits = None
    if robot:
        import yaml
        limits = yaml.safe_load((ROOT / "deploy_mujoco/config/mujoco.yaml").read_text())["tau_limit"]
        hardware_names = list(cfg.asset.real_joint_names)
        actuator_names = [mujoco.mj_id2name(sim.mujoco_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                          for i in range(sim.mujoco_model.nu)]
        effort_limits = np.asarray([limits[hardware_names.index(n)] for n in actuator_names])
        sim.mujoco_model.actuator_ctrllimited[:] = True
        sim.mujoco_model.actuator_ctrlrange[:] = np.column_stack((-effort_limits, effort_limits))
    trace, times, errors, heights, tilts = [], [], [], [], []
    failure = None
    try:
        obs = env.reset()
        with torch.inference_mode():
            for _ in range(20):
                agent.get_action(obs)
            torch.cuda.synchronize()
            body_ids = [mujoco.mj_name2id(sim.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, n)
                        for n in agent.get_meta_data()["selected_body_names"]]
            if min(body_ids) < 0:
                raise ValueError("Robot asset misses selected reference bodies")
            for step in range(round(args.seconds / env.dt)):
                start = time.perf_counter()
                target, action = agent.get_action(obs)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
                if not bool(torch.isfinite(target).all() and torch.isfinite(action).all()):
                    failure = "nonfinite_policy_output"
                    break
                obs = env.step(target, action)
                mujoco.mj_forward(sim.mujoco_model, sim.mujoco_data)
                q = sim.mujoco_data.qpos.copy()
                if not np.isfinite(q).all():
                    failure = "nonfinite_simulation"
                    break
                trace.append(q)
                heights.append(float(q[2]))
                up = sim.mujoco_data.body("pelvis").xmat[8]
                tilts.append(float(np.degrees(np.arccos(np.clip(up, -1, 1)))))
                actual = sim.mujoco_data.xpos[body_ids]
                reference = obs["body_pos_w_future"][0, 0].cpu().numpy()
                errors.append(float(np.linalg.norm(
                    (actual - actual[0]) - (reference - reference[0]), axis=1).mean()))
                if q[2] < .30 or tilts[-1] > 70:
                    failure = "fall_threshold"
                    break
    finally:
        sim.close()
    report = {
        "backend": "mujoco_only", "inference_backend": args.backend, "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "motion": str(motion), "robot_xml": str(cfg.asset.xml_path),
        "torch": torch.__version__, "mujoco": mujoco.__version__,
        "gpu": torch.cuda.get_device_name(), "steps": len(trace),
        "simulated_seconds": len(trace) * env.dt, "policy_dt": env.dt,
        "physics_dt": sim.low_dt, "failure": failure,
        "root_height_min_m": min(heights) if heights else None,
        "root_height_max_m": max(heights) if heights else None,
        "max_pelvis_tilt_deg": max(tilts) if tilts else None,
        "root_relative_body_position_error_mean_m": float(np.mean(errors)) if errors else None,
        "inference_ms_median": float(np.median(times)),
        "inference_ms_p95": float(np.percentile(times, 95)),
        "reference_forcing": True,
        "effort_limit_source": "RoboMimic deploy_mujoco/config/mujoco.yaml" if robot else "official model ctrlrange",
        "scope": "Local offline tracking; no absolute localization or hardware validation.",
    }
    np.savez_compressed(out / "trace.npz", qpos=np.asarray(trace), inference_ms=times,
                        body_error_m=errors, fps=round(1 / env.dt))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
