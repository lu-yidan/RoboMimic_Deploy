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
| **PassiveMode**     | L1 release+R1     | Damping protection mode                                                                  |
| **FixedPose**       | Start             | Position control reset to default joint values                                           |
| **LocoMode**        | R1+A              | Stable walking control mode                                                              |
| **HOST**            | L1+X              | Fall recovery (get-up) controller                                                        |
| **Dance**           | R1+X              | Charleston dance routine                                                                 |
| **KungFu**          | R1+Y              | Martial arts movement                                                                    |
| **KungFu2**         | L1+Y              | Another martial arts movement                                                            |
| **Kick**            | R1+B              | Kicking movement                                                                         |
| **ASAP**            | L1+A              | ASAP locomotion policy                                                                   |
| **BeyondMimic**     | L1+B              | BeyondMimic imitation policy (supports multi-clip NPZ switching)                         |
| **BeyondMimicMJ**   | L1+D-pad DOWN     | Fall-and-get-up imitation policy (MuJoCo-trained, with reference motion tracking)       |
| **StandUpMJ**       | L1+D-pad UP       | Stand-up imitation policy (MuJoCo-trained, with reference motion tracking)              |
| **Score**           | R1+D-pad RIGHT    | Ball-kicking/scoring policy (547-dim obs, 5-frame history, requires onboard LiDAR)      |
| **AMP**             | R2+A              | AMP locomotion policy; R2+D-pad UP → fast mode, R2+D-pad DOWN → slow mode              |
| **SkillCast**       | —                 | Lower body + waist stabilization; upper limbs moved to specific angles (before Mimic)   |
| **SkillCooldown**   | —                 | Lower body + waist balancing; upper limbs reset to default angles (after Mimic)         |


---
## 3. Operation Instructions in Simulation
1. Connect an Xbox controller.
2. Run the simulation program:
```bash
python deploy_mujoco/deploy_mujoco.py
```
3. Press the **Start** button to enter position control mode.
4. Hold **R1 + A** to enter **LocoMode**, then press BACKSPACE in the simulation to make the robot stand. Afterward, use the joystick to control walking.
5. Hold **R1 + X** to enter **Dance** mode—the robot will perform the Charleston. In this mode:
    - Press **Select** at any time to switch to damping protection mode.
    - Hold **R1 + A** to return to walking mode (not recommended).
    - Press **Start** to return to position control mode.
6. The terminal will display a progress bar for the dance. After completion, press **R1 + A** to return to normal walking mode.
7. In LocoMode, pressing **R1 + Y** triggers the KungFu martial arts movement — **use only in simulation**.
8. In LocoMode, pressing **L1 + Y** triggers the KungFu2 martial arts movement — **use only in simulation**.
9. In LocoMode, pressing **R1 + B** triggers the Kick movement — **use only in simulation**.
10. In LocoMode, pressing **L1 + A** enters the ASAP locomotion policy — **use only in simulation**.
11. In LocoMode, pressing **L1 + B** enters the BeyondMimic imitation policy (supports multi-clip NPZ switching) — **use only in simulation**.
12. In LocoMode or FixedPose, pressing **L1 + D-pad DOWN** enters the BeyondMimicMJ fall-and-get-up policy (MuJoCo-trained) — **use only in simulation**.
13. In LocoMode or FixedPose, pressing **L1 + D-pad UP** enters the StandUpMJ stand-up policy (MuJoCo-trained) — **use only in simulation**. BeyondMimicMJ and StandUpMJ can switch directly between each other.
14. In LocoMode, pressing **R1 + D-pad RIGHT** enters the Score ball-kicking policy. Use `--config-name mujoco_score` to load the scene with the ball — **use only in simulation**.
15. In FixedPose or LocoMode, pressing **L1 + X** enters the HOST fall-recovery (get-up) controller — **use only in simulation**.
16. In **FixedPose or LocoMode**, pressing **R2 + A** enters the **AMP** locomotion policy. While in AMP:
    - **R2 + D-pad UP** switches to fast mode (higher speed range).
    - **R2 + D-pad DOWN** switches back to slow mode (default).
    - Press **Start** to return to FixedPose, or **R1 + A** to return to LocoMode.

---
## 4. Real Robot Operation Instructions

1. Power on the robot and suspend it (e.g., with a harness), then hold **L2+R2** to enter debug mode.

2. Run the deploy_real program:
```bash
python deploy_real/deploy_real.py
```
3. Press the **Start** button to enter position control mode.
4. Subsequent operations are largely the same as in simulation.
5. **Score policy (R1+D-pad RIGHT) — additional real-robot steps**: The Score policy depends on onboard LiDAR for real-time ball detection. The perception service must be started before running `deploy_real.py`, publishing ball state via DDS topic `rt/ball_state`. Do **not** activate the Score policy on the real robot without the perception service running.

---
## Important Notes
### 1. Framework Compatibility Notice
The current framework does not natively support deployment on G1 robots equipped with Orin NX platforms. Preliminary analysis suggests compatibility issues with the `unitree_python_sdk` on Orin systems. For onboard Orin deployment, we recommend the following alternative solution:

- Replace with [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) (official C++ SDK)
- Implement a dual-node ROS architecture:
  - **C++ Node**: Handles data transmission between robot and controller
  - **Python Node**: Dedicated to policy inference

### 2. Mimic Policy Reliability Warning
The Mimic policy does not guarantee 100% success rate, particularly on slippery/sandy surfaces. In case of robot instability:
- Press `F1` to activate **PassiveMode** (damping protection)
- Press `Select` to immediately terminate the control program

### 3. Charleston Dance (R1+X) - Stable Policy Notes
Currently the only verified stable policy on physical robots:

⚠️ **Important Precautions**:
- **Palm Removal Recommended**: The original training didn't account for palm collisions (author's G1 lacked palms)
- **Initial/Final Stabilization**: Brief manual stabilization may be required when starting/ending the dance
- **Post-Dance Transition**: While switching to **Locomotion/PositionControl/PassiveMode** is possible, we recommend:
  - First transition to **PositionControl** or **PassiveMode**
  - Provide manual stabilization during transition

### 4. Other Movement Advisories
All other movements are currently **not recommended** for physical robot deployment.

### 5. Strong Recommendation
**Always** master operations in simulation before attempting physical robot deployment.
