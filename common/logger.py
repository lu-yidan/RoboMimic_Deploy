import json
import os
import subprocess
import hashlib
from typing import Optional
import numpy as np
from datetime import datetime
from pathlib import Path


class Logger:
    """Versioned binary logger for FSM policy execution.

    Each call to log() writes one fixed-size float32 record and immediately
    flushes to the OS. This is not an fsync guarantee against power loss.

    Schema 1 record layout (114 float32 values, retained for loading):
        step(1), time_s(1), q(29), dq(29),
        pelvis_pos_w(3), pelvis_quat_wxyz(4),
        ball_pos_b(3), ball_pos_w(3), ball_valid(1),
        target_pos_b(3), target_valid(1),
        vel_cmd(3),
        debug_target_pos_b(3), debug_target_source(1),
        actions(29)

    Schema 2 appends the signals needed for quantitative recovery evidence:
        tau_est(29), tau_cmd_est(29), kps(29), kds(29),
        gravity_ori(3), ang_vel(3), skill_cmd(1)

    ``load`` reads the field layout stored in each log's metadata, so schema 1
    and schema 2 logs remain compatible with schema 3.

    Files written:
        <log_dir>/<YYYYMMDD_HHMMSS>_<tag>.bin   binary frames
        <log_dir>/<YYYYMMDD_HHMMSS>_<tag>.json  metadata
    """

    SCHEMA_VERSION = 3
    LEGACY_FIELDS = [
        ("step",         1),
        ("time_s",       1),
        ("q",           29),
        ("dq",          29),
        ("pelvis_pos_w", 3),
        ("pelvis_quat_w",4),
        ("ball_pos_b",   3),
        ("ball_pos_w",   3),
        ("ball_valid",   1),
        ("target_pos_b", 3),
        ("target_valid", 1),
        ("vel_cmd",      3),
        ("debug_target_pos_b", 3),
        ("debug_target_source", 1),
        ("actions",     29),
    ]
    FIELDS = LEGACY_FIELDS + [
        ("tau_est", 29),
        ("tau_cmd_est", 29),
        ("kps", 29),
        ("kds", 29),
        ("gravity_ori", 3),
        ("ang_vel", 3),
        ("skill_cmd", 1),
    ]
    EXTRA_FIELDS = [
        ("tau_est_valid", 1), ("executed_fsm_state", 1), ("smp_activation_id", 1),
        ("smp_obs", 93), ("smp_raw_action", 29), ("smp_clipped_action", 29),
        ("smp_policy_target", 29), ("smp_prelimit_target", 29),
        ("smp_target_limited", 29), ("smp_warmup_alpha", 1), ("policy_compute_ms", 1),
        ("imu_accel", 3), ("motor_ddq", 29), ("motor_temperature", 58),
        ("state_tick_words", 2), ("command_seq_words", 2), ("state_age_ms", 1),
        ("loop_period_ms", 1), ("runtime_compute_ms", 1), ("remote_raw", 24),
    ]
    FIELDS = FIELDS + EXTRA_FIELDS
    RECORD_DIM: int = sum(n for _, n in FIELDS)

    def __init__(self, log_dir: str, tag: str = "freekick",
                 extra_meta: Optional[dict] = None) -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(log_dir, f"{ts}_{tag}")

        self._bin_path  = base + ".bin"
        self._meta_path = base + ".json"
        self._file      = open(self._bin_path, "wb")
        self._total_steps = 0
        self._events = open(base + ".events.jsonl", "a")
        self._last_fsm = None
        self._policy_meta_written = False
        self._record    = np.empty(self.RECORD_DIM, dtype=np.float32)

        meta: dict = {
            "logger_schema_version": self.SCHEMA_VERSION,
            "tag":         tag,
            "created":     ts,
            "record_dim":  self.RECORD_DIM,
            "dtype":       "float32",
            "fields":      {name: size for name, size in self.FIELDS},
            "control_dt":  0.02,
            "total_steps": 0,
        }
        meta["field_semantics"] = {"actions": "final joint position target, rad",
            "tau_est": "motor firmware estimate, Nm; NaN if unavailable",
            "tau_cmd_est": "host PD estimate Kp*(q_target-q)-Kd*dq; NOT measured torque",
            "state_tick_words": "uint32 tick split into low/high uint16 words",
            "command_seq_words": "uint32 command sequence split into low/high uint16 words",
            "state_age_ms": "local monotonic time since BridgeState reception, not sensor acquisition age",
            "motor_temperature": "29 pairs in SDK order, degrees C",
            "smp_obs": "exact ONNX input, before embedded normalization",
            "missing": "NaN; tau_est_valid=0 when unavailable"}
        root = Path(__file__).resolve().parents[1]
        try:
            meta["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, timeout=2, text=True).strip()
            meta["git_dirty"] = bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, timeout=2, text=True).strip())
        except (OSError, subprocess.SubprocessError):
            meta["git_commit"] = None
        meta["logger_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        from common.utils import FSMStateName
        meta["fsm_state_codes"] = {s.name:s.value for s in FSMStateName}
        meta["joint_order"] = [side + name for side in ("left_", "right_")
            for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
        meta["joint_order"] += ["waist_yaw", "waist_roll", "waist_pitch"]
        meta["joint_order"] += [side + name for side in ("left_", "right_")
            for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
        if extra_meta:
            meta.update(extra_meta)

        with open(self._meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[Logger] writing → {self._bin_path}")

    # ------------------------------------------------------------------

    def log(self, step: int, time_s: float,
            state_cmd, policy_output) -> None:
        """Write one control-step record.  Call after FSM.run()."""
        r = self._record
        offset = 0

        def put(arr, n: int) -> None:
            nonlocal offset
            r[offset : offset + n] = arr
            offset += n

        put([step],                        1)
        put([time_s],                      1)
        put(state_cmd.q,                  29)
        put(state_cmd.dq,                 29)
        put(state_cmd.pelvis_pos_w,        3)
        put(state_cmd.pelvis_quat_w,       4)
        put(state_cmd.ball_pos_b,          3)
        put(state_cmd.ball_pos_w,          3)
        put([float(state_cmd.ball_valid)], 1)
        put(state_cmd.target_pos_b,        3)
        put([float(state_cmd.target_valid)], 1)
        put(state_cmd.vel_cmd,             3)
        put(policy_output.debug_target_pos_b, 3)
        put(policy_output.debug_target_source, 1)
        put(policy_output.actions,        29)
        put(state_cmd.tau_est,             29)
        tau_cmd_est = (
            policy_output.kps * (policy_output.actions - state_cmd.q)
            - policy_output.kds * state_cmd.dq
        )
        put(tau_cmd_est,                   29)
        put(policy_output.kps,             29)
        put(policy_output.kds,             29)
        put(state_cmd.gravity_ori,          3)
        put(state_cmd.ang_vel,              3)
        skill_cmd = getattr(state_cmd.skill_cmd, "value", state_cmd.skill_cmd)
        put([float(skill_cmd)],             1)

        telemetry = getattr(state_cmd, "telemetry", {})
        debug = getattr(policy_output, "recovery_debug", {})
        values = dict(telemetry, **debug)
        values["tau_est_valid"] = float(getattr(state_cmd, "tau_est_valid", False))
        values["executed_fsm_state"] = getattr(policy_output, "executed_fsm_state", -1)
        for name, n in self.EXTRA_FIELDS:
            put(values.get(name, np.full(n, np.nan)), n)
        assert offset == self.RECORD_DIM
        if not self._policy_meta_written and getattr(policy_output, "recovery_metadata", None):
            with open(self._meta_path) as f:
                meta = json.load(f)
            meta["smp_recovery"] = policy_output.recovery_metadata
            with open(self._meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            self._policy_meta_written = True
        if values["executed_fsm_state"] != self._last_fsm:
            self.event("fsm_execution", time_s=time_s, step=step,
                       state=values["executed_fsm_state"])
            self._last_fsm = values["executed_fsm_state"]
        self._file.write(r.tobytes())
        self._file.flush()
        self._total_steps += 1

    def event(self, kind, **data):
        self._events.write(json.dumps({"event": kind, **data}) + "\n")
        self._events.flush()

    def close(self) -> None:
        self._events.close()
        self._file.close()
        with open(self._meta_path) as f:
            meta = json.load(f)
        meta["total_steps"] = self._total_steps
        with open(self._meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Logger] closed — {self._total_steps} frames → {self._bin_path}")

    # ------------------------------------------------------------------

    @staticmethod
    def load(bin_path: str) -> dict:
        """Load a .bin log file.

        Returns a dict mapping field names to numpy arrays:
            scalar fields → shape (T,)
            vector fields → shape (T, N)
        Also includes '_meta' with the raw JSON metadata.
        """
        meta_path = bin_path.replace(".bin", ".json")
        with open(meta_path) as f:
            meta = json.load(f)

        raw = np.fromfile(bin_path, dtype=np.float32)
        dim = int(meta["record_dim"])
        field_mapping = meta.get("fields")
        if not isinstance(field_mapping, dict) or not field_mapping:
            layout = Logger.LEGACY_FIELDS
        else:
            layout = [(str(name), int(size)) for name, size in field_mapping.items()]
        if sum(size for _, size in layout) != dim:
            raise ValueError(
                f"Metadata field layout sums to "
                f"{sum(size for _, size in layout)}, expected record_dim={dim}"
            )
        T   = len(raw) // dim
        if T == 0:
            raise ValueError(f"Log is empty: {bin_path}")

        frames = raw[: T * dim].reshape(T, dim)

        result: dict = {"_meta": meta}
        offset = 0
        for name, size in layout:
            chunk = frames[:, offset : offset + size]
            result[name] = chunk.squeeze(axis=1) if size == 1 else chunk
            offset += size

        return result
