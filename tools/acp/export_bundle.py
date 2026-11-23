#!/usr/bin/env python3
"""Export a real InstinctMJ ACP checkpoint and record native/ONNX parity.

Run this with the Python environment used by InstinctMJ. Nothing is written to
the training repository. This module can also be imported by a training Play
script and its ``export_bundle`` helper called with a loaded env and runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


WRIST_NAMES = tuple(
    f"{side}_wrist_{axis}_joint"
    for side in ("left", "right") for axis in ("roll", "pitch", "yaw")
)
HISTORY_TERMS = (("base_ang_vel", 3), ("projected_gravity", 3),
                 ("joint_pos", 23), ("joint_vel", 23), ("last_action", 23))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inline_onnx_parameters(path: Path) -> None:
    """Make the policy hash cover weights written externally by newer Torch."""
    import onnx
    model = onnx.load(str(path), load_external_data=True)
    onnx.external_data_helper.convert_model_from_external_data(model)
    onnx.save_model(model, str(path), save_as_external_data=False)


def git_source(path: Path) -> dict[str, Any]:
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    return {"path": str(path), "git_head": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain"))}


def read_fixed_joints(path: str | Path) -> dict[str, dict[str, float]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    joints = data.get("fixed_joints", data)
    if set(joints) != set(WRIST_NAMES):
        raise ValueError("fixed_joints must contain exactly the six named G1 wrist joints")
    result = {}
    for name in WRIST_NAMES:
        values = {key: float(joints[name][key]) for key in ("position", "kp", "kd", "effort_limit")}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite fixed joint configuration: {name}")
        if values["position"] != 0.0:
            raise ValueError(f"ACP welded wrist pose is zero: {name}")
        if values["kp"] <= 0 or values["kd"] < 0 or values["effort_limit"] <= 0:
            raise ValueError(f"Invalid fixed wrist gains: {name}")
        result[name] = values
    return result


def _numpy(tensor):
    return tensor.detach().cpu().numpy().copy()


def _action_clip(action_term):
    """The audited ACP task has no clip; do not confuse target and raw clips."""
    clip = getattr(action_term.cfg, "clip", None)
    if clip is None or clip == {}:
        return None
    # mjlab BaseAction.process_actions clips AFTER scale/default offset, in
    # absolute target-radian space. It is not a raw action clamp, and last_action
    # still holds the raw network output. The current schema has no target clip.
    raise ValueError("Custom joint target clipping needs a new runtime contract; audited ACP action_clip is null")


def _history_reset(obs) -> str:
    """Measure reset behavior in the loaded dependency version, without guessing."""
    import numpy as np
    arr = _numpy(obs)
    offset = 0
    for name, width in HISTORY_TERMS:
        history = arr[:, offset:offset + 5 * width].reshape(-1, 5, width)
        if not np.allclose(history, history[:, -1:, :], rtol=0, atol=1e-7):
            raise ValueError(f"Reset {name} history is not repeated; audit this mjlab version before deployment")
        offset += 5 * width
    if not np.allclose(arr[:, 260:375], 0, rtol=0, atol=1e-7):
        raise ValueError("Reset last_action history is not zero")
    return "repeat"


def _apply_goal(env, goal_world):
    if goal_world is None:
        return
    term = env.command_manager.get_term("acp_goal")
    term.target_pos_w[:, 0] = goal_world[0]
    term.target_pos_w[:, 1] = goal_world[1]
    term.target_yaw[:] = math.atan2(math.sin(goal_world[2]), math.cos(goal_world[2]))
    # get_observations() otherwise returns mjlab's cached pre-override goal.
    # Clearing the cache does not push history: compute() defaults to False.
    env.observation_manager._obs_buffer = None


def export_bundle(env, runner, checkpoint: str | Path, output_dir: str | Path,
                  *, fixed_joints: dict, instinct_mj: str | Path,
                  task: str = "Instinct-ACP-Stage3-G1-Play-v1", steps: int = 300,
                  goal_world: tuple[float, float, float] | None = None,
                  atol: float = 1e-5, rtol: float = 1e-4) -> dict[str, Any]:
    """Export loaded ``runner`` in actual environment joint order and validate it.

    ``env`` is the InstinctRlVecEnvWrapper. The caller must load ``checkpoint``
    first. A manifest is published only after every parity check passes. The
    checkpoint itself must prove Stage 3 through ``infos.stage == 3``; a Stage 2
    checkpoint can be exported only with a Stage2 task and remains stage 2.
    """
    import numpy as np
    import onnxruntime as ort
    import torch

    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    if steps < 2:
        raise ValueError("At least two native rollout frames are required")
    if set(fixed_joints) != set(WRIST_NAMES):
        raise ValueError("Pass an explicit, reviewed six-wrist configuration")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not ck.get("has_planner") or "planner_state_dict" not in ck:
        raise ValueError("Checkpoint does not contain a trained planner")
    stage = 3 if isinstance(ck.get("infos"), dict) and ck["infos"].get("stage") == 3 else 2
    if ("Stage3" in task) != (stage == 3):
        raise ValueError(f"Checkpoint proves stage {stage}, inconsistent with task {task}")
    if runner.planner is None:
        raise ValueError("Loaded runner has no planner")
    if Path(runner.resume_path).resolve() != checkpoint:
        raise ValueError("Runner was not loaded from the supplied checkpoint")
    base = env.unwrapped
    if env.num_envs != 1:
        raise ValueError("Use exactly one Play environment for reproducible parity")
    action = base.action_manager.get_term("joint_pos")
    robot = base.scene["robot"]
    ids = action.target_ids.tolist()
    names = [robot.joint_names[i] for i in ids]
    if len(names) != 23 or len(set(names)) != 23 or any("wrist" in n for n in names):
        raise ValueError("Expected exactly 23 distinct non-wrist action joints")
    for term_name in ("joint_pos", "joint_vel"):
        asset = base.cfg.observations["policy"].terms[term_name].params["asset_cfg"]
        obs_ids = asset.joint_ids
        obs_ids = list(range(len(robot.joint_names)))[obs_ids] if isinstance(obs_ids, slice) else list(obs_ids)
        if [robot.joint_names[int(i)] for i in obs_ids] != names:
            raise ValueError(f"{term_name} observation order differs from the action joint order")
    default = _numpy(robot.data.default_joint_pos[0])[ids]
    default_vel = _numpy(robot.data.default_joint_vel[0])[ids]
    if not np.allclose(default_vel, 0, rtol=0, atol=0):
        raise ValueError("Nonzero default joint velocities require a new runtime contract")
    kps, kds, effort_limits = (np.full(len(robot.joint_names), np.nan) for _ in range(3))
    model = base.sim.mj_model
    # Same extraction used by ACPMDP.__init__ in the training repository.
    for actuator in robot.actuators:
        count = len(actuator.target_ids)
        controls = [int(c) for c in actuator.global_ctrl_ids]
        for k, joint_id in enumerate(actuator.target_ids.tolist()):
            kps[joint_id] = float(model.actuator_gainprm[controls[k], 0])
            kds[joint_id] = float(model.actuator_gainprm[controls[count + k], 0])
            global_joint_id = int(robot.indexing.joint_ids[joint_id])
            effort_limits[joint_id] = float(abs(model.jnt_actfrcrange[global_joint_id]).max())
    if not all(np.isfinite(values[ids]).all() for values in (kps, kds, effort_limits)):
        raise ValueError("Could not extract every controlled joint's actual PD gains")
    scales = _numpy(action.scale).reshape(-1, 23)[0]
    scale = float(scales[0]) if np.allclose(scales, scales[0], rtol=0, atol=0) else scales.tolist()
    action_clip = _action_clip(action)
    goal = base.command_manager.get_term("acp_goal")
    target_radius = float(goal.cfg.goal_min_distance if stage == 3 else goal.cfg.min_distance)
    if stage == 3 and not bool(goal.is_goal.all()):
        raise ValueError("Stage3 parity must use the goal-reaching Play configuration")
    runner.eval_mode()
    runner.cfg["onnx_deploy_joint_order"] = False
    obs, _ = env.reset()
    history_reset = _history_reset(obs)
    _apply_goal(base, goal_world)
    obs, extras = env.get_observations()
    output.mkdir(parents=True, exist_ok=True)
    runner.export_policy_onnx(obs, str(output))
    policy = output / "policy.onnx"
    inline_onnx_parameters(policy)
    graph = runner.policy_graph(False).eval()
    session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].name != "planner_obs" or session.get_inputs()[0].shape != [1, 939]:
        raise ValueError("Exported ONNX input contract is not float32[1,939] planner_obs")
    if [out.name for out in session.get_outputs()] != ["actions", "contacts", "occupancy"]:
        raise ValueError("Unexpected exported ONNX output names")
    records: dict[str, list] = {}
    errors = {name: 0.0 for name in ("actions", "contacts", "occupancy", "native_runner_actions")}
    def record(name, value):
        records.setdefault(name, []).append(_numpy(value) if torch.is_tensor(value) else np.asarray(value).copy())
    with torch.inference_mode():
        for _ in range(steps):
            obs, extras = env.get_observations()
            points = extras["observations"]["planner"]
            planner_obs = torch.cat((obs[:, :375], points, obs[:, 465:468]), -1)
            native = graph(planner_obs)
            inputs = _numpy(planner_obs).astype(np.float32)
            onnx = session.run(None, {"planner_obs": inputs})
            for name, expected, actual in zip(("actions", "contacts", "occupancy"), native, onnx):
                expected_np = _numpy(expected)
                np.testing.assert_allclose(actual, expected_np, rtol=rtol, atol=atol, err_msg=name)
                errors[name] = max(errors[name], float(np.max(np.abs(actual - expected_np))))
                record(f"native_{name}", expected)
                record(f"onnx_{name}", actual)
            # Exercise native inference independently of the export graph, using
            # the same captured observations. Do not re-read a noisy observation.
            native_obs, _ = runner.plan_contacts(obs, points, 1.0)
            native_runner = runner.ac.act_inference(runner.normalizers["policy"](native_obs))
            np.testing.assert_allclose(_numpy(native_runner), _numpy(native[0]), rtol=rtol, atol=atol,
                                       err_msg="native runner vs complete policy graph")
            errors["native_runner_actions"] = max(errors["native_runner_actions"],
                                                   float(np.max(np.abs(_numpy(native_runner) - _numpy(native[0])))))
            reference = base.command_manager.get_term("acp_reference_contacts")
            record("planner_obs", planner_obs)
            record("policy_obs", obs)
            record("native_runner_actions", native_runner)
            record("joint_pos", robot.data.joint_pos[:, ids])
            record("joint_vel", robot.data.joint_vel[:, ids])
            record("pelvis_pos_w", reference.base_pos_w)
            record("pelvis_quat_wxyz", reference.base_quat_w)
            record("pelvis_ang_vel_w", reference.base_ang_vel_w)
            record("target_pos_w", goal.target_pos_w)
            record("target_yaw_w", goal.target_yaw)
            record("last_action", base.action_manager.action)
            scanner = base.scene['height_scanner']
            record('terrain_hit_pos_w', scanner.data.hit_pos_w)
            record('terrain_distances', scanner.data.distances)
            record('terrain_frame_pos_w', scanner.data.frame_pos_w)
            _, _, done, _ = env.step(native_runner)
            record("done", done)
            _apply_goal(base, goal_world)
    np.savez_compressed(output / "parity.npz", **{name: np.stack(values) for name, values in records.items()})
    input_contract = {
        "name": "planner_obs", "dtype": "float32", "shape": [1, 939],
        "layout": ["angular_velocity_pelvis[5,3]", "projected_gravity_pelvis[5,3]",
                   "joint_position_minus_default[5,23]", "joint_velocity[5,23]", "last_action[5,23]",
                   "planner_points[11,17,3]", "goal[3]"],
        "history": "term-major, each term oldest to newest at 50 Hz",
        "grid": {"nx": 17, "ny": 11, "resolution": 0.1, "x_min": -0.8, "y_min": -0.5,
                 "order": "y outer, x inner, xyz interleaved", "frame": "pelvis yaw",
                 "z": "pelvis_z - terrain_hit_z - 0.5"},
        "goal": "Rz(-pelvis_yaw)*(target_xy-pelvis_xy)/max(distance,target_radius), wrap(target_yaw-pelvis_yaw)",
        "root": "pelvis", "quaternion_order": "wxyz", "normalization": "inside ONNX",
    }
    versions = {}
    for package in ("torch", "mujoco", "mjlab", "instinct-rl", "onnx", "onnxruntime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    motion_reference = next(sensor for sensor in base.cfg.scene.sensors if sensor.name == "motion_reference")
    dataset = Path(motion_reference.motion_buffers["TerrainMotion"].path).expanduser().resolve()
    dataset_source = {"path": str(dataset), "metadata_sha256": sha256(dataset / "metadata.yaml")}
    for name in ("provenance.json", "contacts_validation.json", "CONTACTS.sha256"):
        if (dataset / name).is_file():
            dataset_source[name + "_sha256"] = sha256(dataset / name)
    manifest = {
        "schema_version": 1, "stage": stage, "planner_obs_dim": 939,
        "policy_sha256": sha256(policy), "policy_storage": "self_contained_onnx", "joint_names": names,
        "default_joint_pos": default.tolist(), "kps": kps[ids].tolist(), "kds": kds[ids].tolist(),
        "effort_limits": effort_limits[ids].tolist(),
        "fixed_joints": fixed_joints, "fixed_joints_source": "explicit deployment configuration; not learned or training actuator gains",
        "action_scale": scale, "action_clip": action_clip, "target_radius": target_radius,
        "control_dt": float(base.step_dt), "history_length": 5, "history_reset": history_reset,
        "default_joint_vel": default_vel.tolist(), "joint_order": "actual environment action target_ids",
        "input_contract": input_contract,
        "output_contract": {"actions": [1, 23], "contacts": [1, 9, 3], "occupancy": [1, 9, 1],
                            "contacts_frame": "pelvis yaw, metric xyz relative to pelvis",
                            "action_target": "default_joint_pos + action_scale * actions"},
        "source": {"instinct_mj": git_source(Path(instinct_mj).expanduser().resolve()),
                   "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                   "checkpoint_iteration": int(ck["iter"]), "checkpoint_stage": stage, "task": task,
                   "exporter_sha256": sha256(Path(__file__).resolve()), "dataset": dataset_source, "versions": versions},
        "parity": {"file": "parity.npz", "sha256": sha256(output / "parity.npz"), "frames": steps,
                   "atol": atol, "rtol": rtol, "max_abs_error": errors,
                   "goal_world_xy_yaw": goal_world, "passed": True},
        "scan_miss_height": float(scanner.cfg.max_distance-scanner.cfg.ray_offset-.5),
        "scan_miss_fill_valid_fraction": 0.75,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instinct-mj", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixed-joints", required=True, type=Path,
                        help="Explicit JSON configuration; review example gains before using hardware")
    parser.add_argument("--dataset", type=Path, default=os.environ.get("ACP_DATASET"),
                        help="ACP TerrainMotion dataset directory (or ACP_DATASET environment variable)")
    parser.add_argument("--task", default="Instinct-ACP-Stage3-G1-Play-v1")
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--goal-world", nargs=3, type=float, metavar=("X", "Y", "YAW"))
    parser.add_argument("--atol", default=1e-5, type=float)
    parser.add_argument("--rtol", default=1e-4, type=float)
    args = parser.parse_args()
    # Validate files before importing or initializing the simulator.
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    source = args.instinct_mj.expanduser().resolve(strict=True)
    fixed = read_fixed_joints(args.fixed_joints)
    sys.path.insert(0, str(source / "src"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    import instinct_mj.tasks  # noqa: F401
    from instinct_mj.tasks.acp.config.g1 import acp_tracking_cfg
    from instinct_mj.scripts.instinct_rl.play import _disable_headless_debug_visualization
    from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls, load_vecenv_cls
    torch.manual_seed(args.seed)
    if args.dataset is not None:
        dataset = Path(args.dataset).expanduser().resolve(strict=True)
        if not (dataset / "metadata.yaml").is_file():
            parser.error(f"Dataset lacks metadata.yaml: {dataset}")
        # The task dataclass factories read this global while constructing the
        # motion reference and terrain. Override BEFORE load_env_cfg invokes
        # __post_init__, which immediately reads metadata.yaml.
        acp_tracking_cfg.MOTION_FOLDER = str(dataset)
        motion_buffer = acp_tracking_cfg.motion_reference_cfg.motion_buffers["TerrainMotion"]
        motion_buffer.path = str(dataset)
        motion_buffer.metadata_yaml = str(dataset / "metadata.yaml")
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_instinct_rl_cfg(args.task)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    _disable_headless_debug_visualization(env_cfg)
    raw_env = load_env_cls(args.task)(cfg=env_cfg, device=args.device)
    env = load_vecenv_cls(args.task)(raw_env, policy_group=agent_cfg.policy_observation_group,
                                   critic_group=agent_cfg.critic_observation_group)
    try:
        runner = load_runner_cls(args.task)(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(str(checkpoint))
        manifest = export_bundle(env, runner, checkpoint, args.output, fixed_joints=fixed,
                                 instinct_mj=source, task=args.task, steps=args.steps,
                                 goal_world=args.goal_world, atol=args.atol, rtol=args.rtol)
        print(json.dumps({"output": str(args.output.resolve()), "stage": manifest["stage"],
                          "policy_sha256": manifest["policy_sha256"], "parity": manifest["parity"]}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
