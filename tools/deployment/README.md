# Offline AMP/SMP branch replay

`replay_branches.py` compares bridge Python `PolicyRuntime.step`, `FSM.run`,
Passive/FixedPose and actual AMP/SMP ONNX inference in separate processes.
It never calls runtime/FSM constructors or creates DDS participants/publishers.
ACP, PHP and other learned policies are excluded. Their button requests fail.

```bash
python tools/deployment/replay_branches.py \
  --baseline-root ../RoboMimic_Deploy_under_table_review \
  --input /absolute/path/state-input.npz --initial-state PASSIVE \
  --amp-profile amp_0309_1 --smp-profile v34_balanced4_hard19000 \
  --amp-model /absolute/path/amp_0309_1.onnx \
  --smp-model /absolute/path/smp_v34_balanced4_hard19000_93d.onnx \
  --output-dir /absolute/path/replay-result
```

Profiles and exact model bytes are explicit and SHA-256 checked against each
checkout. Baseline AMP supports its original `amp_0309_1` profile. No source or
configuration in either checkout is rewritten. JSON contains 29 joint targets,
Kp/Kd, current/executed FSM state, enter calls, damping, receipt-age timeout,
sequence, and every differing frame with affected motor indices.
The output directory must be new or empty so reruns preserve existing evidence
and cannot overwrite the input log.

Inputs may be a logger `.bin` plus its `.json`, or NPZ arrays `q(N,29)`,
`dq(N,29)`, `imu_quat_wxyz(N,4)`, `imu_gyro(N,3)`, `remote_raw(N,24)`,
`time_s(N)`, `tick(N)`, `state_age_ms(N)`. NPZ also requires scalar
`provenance_json` with `kind: recorded_log` or `synthetic_diagnostic`.
Missing telemetry fails; it is never replaced by invented observations.
Reception age over `--stale-ms` (default 200) is mapped to the subscriber's
`tick=0` stale-state result. DDS polling and C++ command timeout are not run.

Policy hidden history is explicitly reset at the selected initial state.
Logs missing the original entry/history can support a comparison of the same
recorded segment with this shared initialization; they cannot reproduce the
original motor commands exactly. No result verifies physics, direct Python
hardware deployment, or a historical robot system. Synthetic results say
`ready_for_real_log`, never `hardware_baseline_verified`.

Use `--make-diagnostic /path/input.npz --scenario nominal` for an explicitly
synthetic ordinary AMP/SMP enter/run/exit sequence. Separate `f1`, `stale`,
`start_y` and `a_y` scenarios isolate priority changes and avoid attributing
later history divergence to a new bug. `mixed` is the regression fixture.
