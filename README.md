# lyd-deploy

Unified Unitree G1 (29-DoF) deployment, based on [RoboNaldo Deploy](https://github.com/OpenDriveLab/RoboNaldo_Deploy). See the [deployment guide](README_zh.md) for setup, simulation, controls and validation.

Edit **`config/deploy.yaml`**, then restart:

- **ScaleBFM:** optional, disabled by default. Enabling requires the external ScaleBFM checkout, model artifacts and CUDA runtime.
- **AMP:** `amp_0309_1`, `stratified_guidance_60000`, or `stratified_single_60000`.
- **SMP:** `path_a6_9999`, `a6_no_alpha_9999`, or `v34_balanced4_hard19000` (default). Each profile retains its model hash and torque limits.

MuJoCo, Python DDS and the C++ bridge policy process share the same selection file. Table and free-box/lid scenes are configured separately in `deploy_mujoco/config/`.

```bash
git clone --single-branch --branch lyd-deploy --depth 1 git@github.com:lu-yidan/RoboMimic_Deploy.git
```

This branch starts from a consolidated snapshot. Historical experiments are not part of its Git ancestry. Upstream licenses and attribution are retained.
