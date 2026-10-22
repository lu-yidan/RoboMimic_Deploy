"""Export a dependency-scoped, relocatable simulation candidate; preserve audits."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys
import torch
from deploy_mujoco.run_geometry_full_cycle import load_verified
from tools.continue_vector_full_cycle_training import sha
from training.full_cycle_residual_env import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    base, observer, model, manifest, summary = load_verified(args.experiment, args.case)
    try:
        audit = json.loads((args.experiment/'independent_training_audit.json').read_text())
        row = next(r for r in audit['trajectories'] if r['label']=='after' and r['case']==args.case)
        golden = args.experiment/f'after_{args.case}.npz'
        assert row['passed'] and row['independently_accepted'] and sha(golden)==row['trajectory_sha256']
        observation, _ = observer.reset(seed=20261022, options={'case':args.case})
        model.predict(observation, deterministic=True)
        sources = {}
        for module in list(sys.modules.values()):
            source = getattr(module, '__file__', None)
            if not source:
                continue
            source = Path(source).resolve()
            if source.suffix != '.py' or not source.is_relative_to(ROOT):
                continue
            relative = source.relative_to(ROOT)
            if relative.parts[0] in ('training', 'tools', 'policy', 'deploy_mujoco'):
                sources[str(relative)] = source
        sources['deploy_mujoco/run_packaged_geometry.py'] = ROOT/'deploy_mujoco/run_packaged_geometry.py'
        args.output.mkdir(parents=True, exist_ok=False)
        for relative, source in sources.items():
            destination = args.output/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        bundle = args.output/'outputs/wall_support_bundle_pd15_v2'
        shutil.copytree(base.env.bundle, bundle)
        reference = args.output/'reference'
        reference.mkdir()
        for name in ('plan.npz', 'manifest.json', 'audit.json'):
            shutil.copy2(Path(manifest['reference']).parent/name, reference/name)
        evidence = args.output/'evidence'
        evidence.mkdir()
        for name in ('manifest.json', 'summary.json', 'independent_training_audit.json'):
            shutil.copy2(args.experiment/name, evidence/name)
        shutil.copy2(golden, evidence/'golden_trajectory.npz')
        shutil.copy2(args.experiment/'policy.zip', args.output/'policy.zip')
        versions = {name:metadata.version(name) for name in ('numpy', 'scipy', 'torch', 'mujoco', 'gymnasium', 'stable_baselines3', 'osqp')}
        (args.output/'requirements-runtime.txt').write_text('\n'.join(name+'=='+version for name,version in versions.items())+'\n')
        (args.output/'README_zh.md').write_text('''# G1 钩爪攀爬仿真候选包

此包只复现一个已审计工况。包含黑色钩爪、横杆场景、参考轨迹、148维观测策略和必要源码；不连接真机。
这是候选包，完整二阶段选模结果另行记录。控制器依赖仿真真实接触，策略读取真实杆偏移，并非视觉自主攀爬。

进入本目录，用匹配依赖的 Python 运行：

```powershell
python -m deploy_mujoco.run_packaged_geometry --verify-only
python -m deploy_mujoco.run_packaged_geometry --output replay
python -m deploy_mujoco.run_packaged_geometry --output viewer_replay --viewer
```

输出目录必须不存在。头部界面关闭会结束运行，不能计作完整成功。包内脚本核对所有文件和主要依赖版本；完整运行后将七组数组与原审计轨迹逐项精确比较。
同机独立路径复现通过后，仍不能推断其他操作系统、硬件或浮点库上的逐位一致。原始审计保持原样放在 evidence，绝对路径字段仅作来源记录；运行路径由 release_manifest.json 的相对路径定义。
''', encoding='utf-8')
        files = {str(path.relative_to(args.output)):sha(path) for path in args.output.rglob('*') if path.is_file()}
        result = dict(format='g1-climb-simulation-candidate-v1', simulation_only=True, selection_status='candidate_pending_complete_second_stage',
                      case=args.case, offsets=manifest['evaluation_cases'][args.case], geometry_mode=manifest['geometry_mode'],
                      normal_bias_y_deg=manifest['normal_bias_y_deg'], force_history_memory=manifest['force_history_memory'],
                      reference='reference/plan.npz', checkpoint='policy.zip', golden_trajectory='evidence/golden_trajectory.npz',
                      original_experiment=str(args.experiment.resolve()), original_audit_sha256=sha(args.experiment/'independent_training_audit.json'),
                      contract=manifest['contract'], observation_contract=manifest['observation_contract'],
                      runtime_versions=versions, files=files, exporter_sha256=sha(__file__),
                      scope='Relocatable simulation candidate with unchanged physical scene/reference/actor and original audit evidence. Validate by independent full-cycle package replay before claiming packaging success.')
        (args.output/'release_manifest.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(output=str(args.output), files=len(files), source_files=len(sources), bytes=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()))))
    finally:
        observer.close()


if __name__ == '__main__':
    main()
