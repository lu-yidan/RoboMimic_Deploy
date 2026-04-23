<div align="center">
  <h1 align="center">RoboMimic Deploy</h1>
  <p align="center">
    <span> 🌎English </span> | <a href="README_zh.md"> 🇨🇳中文 </a>
  </p>
</div>

<p align="center">
  <strong>​RoboMimic Deploy​​ is a multi-policy robot deployment framework based on a state-switching mechanism. Currently, the included policies are designed for the ​​Unitree G1 robot (29-DoF)​​.</strong> 
</p>

## Preface

- **​This deployment framework is only applicable to G1 robots with a 3-DOF waist. If a waist fixing bracket is installed, it must be unlocked according to the official tutorial before this framework can be used normally.​​**

- **It is recommended to remove the hands, as dance movements may cause interference.​**
  
- **When deploying real robots, if something goes wrong, it's probably the policy's fault—not your hardware. Don't waste time second-guessing your robot's physical setup.**

- **[video instruction](https://www.bilibili.com/video/BV1VTKHzSE6C/?vd_source=713b35f59bdf42930757aea07a44e7cb#reply114743994027967)**

## Installation and Configuration

## 1. Create a Virtual Environment

It is recommended to run training or deployment programs in a virtual environment. We suggest using Conda to create one.

### 1.1 Create a New Environment

Use the following command to create a virtual environment:
```bash
conda create -n robomimic python=3.8
```

### 1.2 Activate the Virtual Environment

```bash
conda activate robomimic
```

---

## 2. Install Dependencies

### 2.1 Install PyTorch
PyTorch is a neural network computation framework used for model training and inference. Install it with the following command:
```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2.2 Install RoboMimic_Deploy

#### 2.2.1 Download
Clone the repository via git:

```bash
git clone https://github.com/ccrpRepo/RoboMimic_Deploy.git
```

#### 2.2.2 Install Components

Navigate to the directory and install:
```bash
cd RoboMimic_Deploy
pip install -r requirements.txt
```

#### 2.2.3 Install unitree_sdk2_python

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```
---
## Running the Code

## 1. Run Mujoco Simulation

Standard scene:
```bash
python deploy_mujoco/deploy_mujoco.py
```

Score scene (includes ball, required for Score policy):
```bash
python deploy_mujoco/deploy_mujoco.py --config-name mujoco_score
```

## 2. Policy Descriptions
| Mode Name           | Trigger Keys      | Description                                                                              |
|---------------------|-------------------|------------------------------------------------------------------------------------------|
| **PassiveMode**     | L1                | Damping protection mode                                                                  |
| **FixedPose**       | Start             | Position control reset to default joint values                                           |
| **LocoMode**        | B                 | Stable walking control mode                                                              |
| **AMP**             | A                 | AMP locomotion policy (enters run mode by default)                                      |
| **Score**           | R1                | Ball-kicking/scoring policy (547-dim obs, 5-frame history, requires onboard LiDAR)      |
| **BeyondMimicMJ**   | D-pad DOWN        | Fall-and-get-up imitation policy (MuJoCo-trained, with reference motion tracking)       |
| **StandUpMJ**       | D-pad UP          | Stand-up imitation policy (MuJoCo-trained, with reference motion tracking)              |
| **Pinocchio1.6MJ**  | R2                | MuJoCo imitation policy backed by `g1_result_pinocchio_1_6_mj.yaml`                     |
| **BeyondMimic**     | —                 | Still kept in the repo, but not bound to a default controller shortcut                  |


---
## 3. Operation Instructions in Simulation
1. Connect an Xbox controller.
2. Run the simulation program:
```bash
python deploy_mujoco/deploy_mujoco.py
```
3. Press the **Start** button to enter position control mode.
4. Press **B** to enter **LocoMode**, then press BACKSPACE in the simulation to make the robot stand. Afterward, use the joystick to control walking.
5. The simplified single-button policy map is:
   - **A** -> AMP (run mode)
   - **B** -> LocoMode
   - **D-pad DOWN** -> BeyondMimicMJ
   - **D-pad UP** -> StandUpMJ
   - **R2** -> Pinocchio1.6MJ
   - **R1** -> Score
6. Use `--config-name mujoco_score` when entering **Score** so the simulation loads the ball scene.
7. **BeyondMimicMJ / StandUpMJ / Pinocchio1.6MJ** can switch directly between each other with the same single-button bindings.
8. Press **L1** at any time for damping protection, or **Start** to return to **FixedPose**.
9. **BeyondMimic** is still present in the repository, but it is intentionally left without a default controller shortcut. Legacy policies such as Dance / Kick / KungFu / ASAP / HOST are removed from the default deploy switching path.

---
## 4. Real Robot Operation Instructions

1. Power on the robot and suspend it (e.g., with a harness), then hold **L2+R2** to enter debug mode.

2. Run the deploy_real program:
```bash
python deploy_real/deploy_real.py
```
3. Press the **Start** button to enter position control mode.
4. The same simplified single-button mapping is used on the real robot: **A / B / D-pad DOWN / D-pad UP / R2 / R1**.
5. **Score policy (R1) — additional real-robot steps**: The Score policy depends on onboard LiDAR for real-time ball detection. The perception service must be started before running `deploy_real.py`, publishing ball state via DDS topic `rt/ball_state`. Do **not** activate the Score policy on the real robot without the perception service running.

6. If you are deploying on an onboard Orin/G1 computer, prefer the bridge-based deployment flow under `bridge/` instead of running `deploy_real.py` directly. See:
   - `bridge/README.md`
   - `bridge/VALIDATION.md`

   Python prototype:
   ```bash
   python bridge/python/deploy_bridge_py.py
   python bridge/python/deploy_policy.py
   ```

   C++ bridge flow:
   ```bash
   cmake -S bridge -B bridge/build
   cmake --build bridge/build -j2
   BRIDGE_NETWORK_INTERFACE=eth0 bridge/build/cpp_bridge_main
   python bridge/python/deploy_policy.py
   ```

---
## Important Notes
### 1. Framework Compatibility Notice
The current framework does not natively support deployment on G1 robots equipped with Orin NX platforms. Preliminary analysis suggests compatibility issues with the `unitree_python_sdk` on Orin systems. For onboard Orin deployment, we recommend the following alternative solution:

- Replace with [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) (official C++ SDK)
- Implement a dual-node ROS architecture:
  - **C++ Node**: Handles data transmission between robot and controller
  - **Python Node**: Dedicated to policy inference
- This repository now includes a minimal bridge prototype under `bridge/` that can serve as a reference implementation for that split.

### 2. Mimic Policy Reliability Warning
The Mimic policy does not guarantee 100% success rate, particularly on slippery/sandy surfaces. In case of robot instability:
- Press `F1` to activate **PassiveMode** (damping protection)
- Press `Select` to immediately terminate the control program

### 3. Notes For The Retained Policies
The default deploy entry points are now limited to `LocoMode / AMP / Score / BeyondMimicMJ / StandUpMJ / Pinocchio1.6MJ`.

⚠️ **Important Precautions**:
- **Palm Removal Recommended**: Some original training setups did not account for palm collisions (the author's G1 had no palms)
- **Prefer Safe Transition Paths**: Before and after testing a retained policy, prefer returning to **FixedPose** or **PassiveMode**
- **Use Human Supervision**: For first-time real-robot validation, keep the robot suspended and maintain manual supervision throughout

### 4. Other Movement Advisories
Other historical policy implementations may still exist in the repository, but they are no longer bound to the default deploy shortcuts and are not recommended for direct real-robot use.

### 5. Strong Recommendation
**Always** master operations in simulation before attempting physical robot deployment.
