# ACP deployment bundle export

Use the Python environment that trains the checked-out InstinctMJ ACP code. A
Stage 3 checkpoint must contain `infos.stage == 3`, `has_planner == True`, and
`planner_state_dict`. A Stage 1 or Stage 2 checkpoint cannot produce a Stage 3
manifest. The script never generates replacement or random weights.

```bash
python tools/acp/export_bundle.py \
  --instinct-mj /root/InstinctMJ_acp_g1 \
  --dataset /root/autodl-tmp/acp-g1/data/php_flat \
  --task Instinct-ACP-Stage3-Flat-G1-Play-v1 \
  --checkpoint /absolute/path/to/stage3/model_10000.pt \
  --fixed-joints /absolute/path/to/reviewed-wrists.json \
  --output /absolute/path/to/new-empty-bundle \
  --device cuda:0 --steps 300
```

`ACP_DATASET` can supply the dataset path instead of `--dataset`. Dataset
selection happens before task configuration construction. Optional
`--goal-world X Y YAW` fixes a world-frame target for the native parity rollout.

Review `fixed_joints.example.json` before copying it. Its zero wrist angles match
the welded ACP asset; its gains and 5 Nm limit are explicit deployment
suggestions, since the training model has no wrist actuators.

The exporter records actual environment joint names, default angles, gains,
effort limits, action scale, target radius, control period, observed reset
history, source hashes and input/output contracts. `policy.onnx` includes
observation normalization, contact planning and tracker scan extraction.
`planner_obs` is raw `float32[1,939]`. The manifest retains the actual environment
joint order; the runtime must map names to hardware slots.

`parity.npz` contains every 939-dimensional input, native complete-graph outputs,
independent native-runner actions, ONNX outputs, joint states, pelvis pose,
target and done flags. A manifest is written only after all recorded frames pass
parity checks. Native actions are raw and unclipped in the audited ACP task;
custom absolute joint-target clipping requires a new contract and is rejected.

To call from an existing Play script, import `export_bundle` from this file and
pass its loaded vector environment, loaded runner, checkpoint path, output
directory, explicit fixed-wrist dictionary, source checkout and task ID.
