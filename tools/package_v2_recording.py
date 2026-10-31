"""Export one accepted record for portable playback, preserving original audits.

This is an evidence/replay package, not a portable policy-training environment.
The compiled scene is checked against the source model before redistribution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import mujoco
import numpy as np
from deploy_mujoco.replay_verified_climb import load_recording


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();out=a.output.resolve();root=Path(__file__).resolve().parents[1]
    env,records,ends=load_recording(run)
    m=json.loads((run/'manifest.json').read_text());assert m['acceptance']=='physical-v2'
    out.mkdir(parents=True,exist_ok=False);(out/'evidence').mkdir()
    try:
        mujoco.mj_saveModel(env.model,str(out/'model.mjb'))
        restored=mujoco.MjModel.from_binary_path(str(out/'model.mjb'))
        arrays={name:bool(np.array_equal(value,getattr(restored,name))) for name in dir(env.model)
                if isinstance((value:=getattr(env.model,name)),np.ndarray)}
        assert arrays and all(arrays.values()),[k for k,v in arrays.items() if not v]
        d=mujoco.MjData(restored);pose_checks=[]
        for i in (0,len(records['qpos'])//2,len(records['qpos'])-1):
            env.data.qpos[:]=records['qpos'][i];d.qpos[:]=records['qpos'][i]
            mujoco.mj_kinematics(env.model,env.data);mujoco.mj_kinematics(restored,d)
            pose_checks.append(bool(np.array_equal(env.data.xpos,d.xpos) and np.array_equal(env.data.geom_xpos,d.geom_xpos)))
        assert all(pose_checks)
        shutil.copy2(env.bundle/'LICENSE',out/'LICENSE.Unitree')
    finally:env.close()
    for name in ('trajectory.npz','reference.npz'):shutil.copy2(run/name,out/name)
    for name in ('manifest.json','summary.json','independent_audit.json','policy_transfer_audit.json','v2_multicycle_audit.json','scene_variant.json'):
        shutil.copy2(run/name,out/'evidence'/name)
    shutil.copy2(root/'deploy_mujoco/portable_recording_viewer.py',out/'replay.py')
    (out/'requirements.txt').write_text('numpy=='+np.__version__+'\nmujoco=='+mujoco.__version__+'\n',encoding='utf8')
    (out/'README.md').write_text('''# G1 黑色钩爪：已审计连续八周期回放

这是记录回放包：显示已通过审计的 MuJoCo 姿态记录，不执行策略推理、不重新计算动力学，也不连接真机。来源使用冻结 PPO、脚本参考和仿真真值。原始审计文件保持不变，其中历史绝对路径仅用于来源记录，回放无需这些路径。

在任意目录解压，使用 Python 3.12。在 PowerShell 中运行：

```powershell
python -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.venv\\Scripts\\python.exe replay.py --verify-only
.venv\\Scripts\\python.exe replay.py --speed 8
```

空格暂停，R 重播，鼠标调整视角。`--loop` 循环播放。可先只验证文件与轨迹，不打开窗口。Linux 的 Python 路径改为 `.venv/bin/python`；图形回放需要可用桌面/OpenGL。

`model.mjb` 是经结构数组和运动学一致性核验的编译场景，已包含网格；使用 requirements 中固定的 MuJoCo 版本。`trajectory.npz` 与原始通过审计的记录字节一致；审计文件在 evidence 中。

模型来源：Unitree G1，许可见 LICENSE.Unitree；研究框架源自 ccrpRepo/RoboMimic_Deploy，当前研究分支为 zuolian11/g1-climbing-research。此包不替代完整研究仓库、训练资产或审计源码链。
''',encoding='utf8')
    manifest=dict(case=m['case'],cycles=len(ends),horizon_s=float(records['time_s'][-1]),root_rise_m=float(records['qpos'][-1,2]-records['qpos'][0,2]),
        mujoco_version=mujoco.__version__,source_run=str(run),source_bundle_manifest_sha256=m['bundle_manifest_sha256'],
        source_scene_variant_sha256=sha(run/'scene_variant.json'),source_audit_sha256=sha(run/'v2_multicycle_audit.json'),
        compiled_model_array_checks=arrays,kinematic_pose_checks=pose_checks,exporter_sha256=sha(__file__),
        scope='Portable recorded kinematic replay only, not a fresh policy evaluation or complete reproducibility archive.')
    manifest['files']={str(f.relative_to(out)).replace('\\','/'):sha(f) for f in out.rglob('*') if f.is_file()}
    (out/'package.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in out.rglob('*'):
            if file.is_file():z.write(file,Path(out.name)/file.relative_to(out))
    print(json.dumps(dict(package=str(out),archive=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size,model_arrays_exact=len(arrays))),flush=True)


if __name__=='__main__':main()
