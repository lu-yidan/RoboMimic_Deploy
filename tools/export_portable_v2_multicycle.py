"""Export a fresh-physics v2 package with a frozen accepted golden trajectory."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys
from deploy_mujoco.run_portable_v2_multicycle import create_runtime,sha
from deploy_mujoco.replay_verified_climb import load_recording


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();out=a.output.resolve();root=Path(__file__).resolve().parents[1]
    verified,_,ends=load_recording(run);verified.close()
    m=json.loads((run/'manifest.json').read_text());assert m['acceptance']=='physical-v2'
    if m.get('contact_admission') or m.get('timed_loading') or m.get('scene_compiler'):
        raise ValueError('Experimental controllers and compiled variants are not implemented by this frozen baseline exporter')
    parent=Path(m['parent_experiment']);composite=json.loads((parent/'manifest.json').read_text())
    assert sha(parent/'policy.zip')==m['checkpoint_sha256']
    assert sha(composite['prefix_checkpoint'])==m['prefix_checkpoint_sha256']
    description=dict(format='g1-physical-v2-multicycle',simulation_only=True,case=m['case'],offsets=m['offsets'],cycles=len(ends),
        geometry_mode=composite['geometry_mode'],normal_bias_y_deg=composite['normal_bias_y_deg'],
        force_history_memory=composite['force_history_memory'],fade_seconds=m['fade_seconds'],composite=composite,
        reference=str(Path(m['plan_path'])),checkpoint=str(parent/'policy.zip'))
    base,observer,actor,raw,_,_,_=create_runtime(root,description)
    try:
        actor.predict(raw,deterministic=True)
        sources={}
        # Include the independent endpoint auditor used only after the rollout.
        import tools.audit_multirung_cycle
        for module in list(sys.modules.values()):
            filename=getattr(module,'__file__',None)
            if not filename:continue
            path=Path(filename).resolve()
            if path.suffix=='.py' and path.is_relative_to(root):
                relative=path.relative_to(root)
                if relative.parts[0] in ('training','tools','policy','deploy_mujoco'):sources[relative.as_posix()]=path
        # Imported execution code must match the completed experiment for every
        # training module present in its frozen source snapshot.
        for rel,digest in m['sources'].items():
            if rel.startswith('training/'):
                assert sha(run/'source_snapshot'/rel)==digest==sha(root/rel)
        out.mkdir(parents=True,exist_ok=False)
        for relative,path in sources.items():
            target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        shutil.copytree(base.env.bundle,out/'outputs/wall_support_bundle_pd15_v2')
        reference=out/'reference';reference.mkdir()
        for name in ('plan.npz','manifest.json','audit.json'):shutil.copy2(Path(m['plan_path']).parent/name,reference/name)
        evidence=out/'evidence';evidence.mkdir()
        for name in ('manifest.json','summary.json','independent_audit.json','policy_transfer_audit.json','v2_multicycle_audit.json','endpoint_samples.json','scene_variant.json'):
            shutil.copy2(run/name,evidence/name)
        shutil.copy2(run/'trajectory.npz',evidence/'golden_trajectory.npz')
        shutil.copy2(parent/'policy.zip',out/'policy.zip');shutil.copy2(composite['prefix_checkpoint'],out/'prefix_policy.zip')
        versions={name:metadata.version(name) for name in ('numpy','scipy','torch','mujoco','gymnasium','stable_baselines3','osqp','PyYAML')}
        (out/'requirements-runtime.txt').write_text('\n'.join(name+'=='+version for name,version in versions.items())+'\n',encoding='utf8')
        (out/'README_zh.md').write_text('''# G1 v2 连续攀爬：可搬迁仿真包

本包执行新的 MuJoCo 动力学和冻结策略推理；与只显示姿态的录像回放不同。指定工况及周期数见 release_manifest.json，场景、参考、模型、前缀策略和黄金记录均在包内。历史 evidence 文件里的绝对路径只用于来源追溯，不参与运行。

在本目录安装 requirements-runtime.txt 后运行：

```powershell
python -m deploy_mujoco.run_portable_v2_multicycle --verify-only
python -m deploy_mujoco.run_portable_v2_multicycle --output new_run
python -m deploy_mujoco.run_portable_v2_multicycle --output new_viewer_run --viewer
```

使用 Python 3.12；输出目录必须不存在。--verify-only 只检查资产和依赖，不证明完成动作。完整无窗口运行结束后，必须检查 reproduction_audit.json 的 passed、exact_arrays 和 endpoint_contact_records_exact。提前关窗不能算通过。

固定脚本参考、仿真真值杆位、理想接触反馈和动作校准仍存在。本轮没有重训，也不连接真机。本包尚须完成搬迁后的全周期复现才能称为已验证；仅导出成功不算验证。

模型来源 Unitree，许可在 outputs/wall_support_bundle_pd15_v2/LICENSE。研究框架源自 ccrpRepo/RoboMimic_Deploy；协作仓库为 zuolian11/g1-climbing-research。保留各来源原许可，未赋予整个混合快照新许可证。
''',encoding='utf8')
        description.update(reference='reference/plan.npz',checkpoint='policy.zip',golden_trajectory='evidence/golden_trajectory.npz',
            composite={**composite,'prefix_checkpoint':'prefix_policy.zip'},runtime_versions=versions,
            original_run=str(run),original_audit_sha256=sha(run/'v2_multicycle_audit.json'),
            original_trajectory_sha256=sha(run/'trajectory.npz'),exporter_sha256=sha(__file__),
            validation_status='exported_not_yet_physically_reproduced',
            scope='Relocatable package for new continuous MuJoCo dynamics and frozen composite policy. No training or hardware. A full exact replay remains required after relocation.')
        description['files']={f.relative_to(out).as_posix():sha(f) for f in out.rglob('*') if f.is_file()}
        (out/'release_manifest.json').write_text(json.dumps(description,indent=2),encoding='utf8')
        print(json.dumps(dict(exported=True,output=str(out),case=m['case'],cycles=len(ends),files=len(description['files']),source_files=len(sources))),flush=True)
    finally:observer.close()


if __name__=='__main__':main()
