# Climb deployment adapter

This module follows the `lu_pub` framework's `StateAndCmd` / `PolicyOutput` and
`FSMState.enter/run/exit/checkChange` interfaces. The branch starts at upstream
commit `0dd6ab4899068a9987ca596a1c32cb1c94f22ec4`.

## Implemented interface

- A 50 Hz actor receives `float32 obs[1,100]` and returns `float32 actions[1,29]`.
- Observation order: body gravity (3), body angular velocity (3), joint position
  relative to the configured default (29), scaled joint velocity (29), previous
  clipped policy action (29), target position in the pelvis frame (3), target
  rod axis in that frame (3), and target validity (1).
- Actions are clipped to [-1,1], converted to joint position targets using the
  declared default/scale, then bounded by joint limits. Joint order, gains,
  limits and scales live in `config/climb.json`; training must use the same file
  and `contract.py`. These borrowed gains are not validated climbing gains.
- `PolicyOutput` carries position targets, kp and kd. No QP feedforward torque,
  synthetic wrist/ankle force, world base position or privileged state is added
  to this first contract. Force-based research requires a separately versioned
  observation contract with an actual measurement source.
- `StateAndCmd.climb_target_*` must be explicitly populated by a producer.
  Missing targets, malformed outputs or inference errors latch a fault and
  return the framework's damping output. Damping is not a proved physical
  recovery during climbing.

`FSM(..., profile="climb", climb_options={...})` loads only Passive, FixedPose
and Climb. Existing default `profile="full"` retains the original policy set.
`CMD_CLIMB` is accepted from FixedPose; no new real-robot button binding is enabled.
Use the actual state fields and frequency in a deployment producer; adding the
adapter alone does not connect target perception, synchronized sensors or physics.

## Verification without model weights or hardware

From the repository root, using a separate Python 3.12 environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-climb-check.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_climb_contract.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_climb_adapter.py" -v
.\.venv\Scripts\python.exe tools/check_climb_interface.py
```

The check-only dependencies support the nine contract/adapter tests. To include
the three MuJoCo environment tests, install `requirements-climb-env.txt` and
import the scene first, following `training/README.md`. All 12 tests passed on
2026-10-09. A 4096-step PPO development actor has also been exported and executed
through ONNX Runtime and this adapter in simulation.

The latter deliberately injects a zero-action test actor. It performs no
`mj_step`, sends no robot commands and is not a trained-policy demonstration.

## Loading an exported model

Export one static-batch ONNX input `obs` with shape `[1,100]`, and one output
`actions` with shape `[1,29]`, both float32. Save a JSON sidecar with
`contract_sha256` (the canonical contract fingerprint) and `model_sha256` (the
actual ONNX bytes). Pass `model_path` and `metadata_path` through `climb_options`.
ONNX Runtime is imported only when loading a model; no silent dummy actor is
used if weights are missing. Training must reproduce clipping/history and the
PD position-action interpretation, including the physics/control timing.

The first mounted-support actor uses **its exported `contract.json`**, not the
standing defaults in `policy/climb/config/climb.json`. The mounted contract
fingerprint is `c563e72f5c2f476a72da558de08442fc9c9f030c8887d1c062e0f9c1ed116ef3`.
Pass all three paths explicitly:

```python
from pathlib import Path
from policy.climb.Climb import Climb

export = Path("outputs/climb_ppo_smoke_v1/export")
policy = Climb(state_cmd, policy_output,
    config_path=export / "contract.json",
    model_path=export / "policy.onnx",
    metadata_path=export / "metadata.json")
```

The caller must provide synchronized state and a valid target each policy step.
The current simulation starts already positioned at the rods. Neither this
example nor the generic FixedPose transition solves approaching/mounting a rod.
Do not interpret an API-compatible actor as a verified hardware behavior.

## Remaining integration work

The mounted-wall MuJoCo scene and Gymnasium environment are now connected.
`training/README.md` describes the deterministic support-pose task and measured
results. The old 500 Hz QP controller is absent from the 50 Hz position-policy
environment. One short training run and export equivalence passed; the actor
still fails the declared support-pose success criterion. Rod transfer, verified
load bearing, randomized evaluation, parallel sampling, real perception and
hardware deployment remain unfinished.

This checkout currently contains sparse source/doc files; upstream mesh and
policy-weight assets have not been checked out. The separate local research
scene was imported into ignored `assets/climb_mounted`. Use a separate compatible
environment rather than installing the old pinned requirements into the
existing research environment. No deployment process has been launched.
