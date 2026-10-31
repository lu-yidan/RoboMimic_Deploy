"""Bundle an immutable live simulator package with its later reproduction proof."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
from tools.validate_swing_reference import read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True)
    p.add_argument('--reproduction',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();package=a.package.resolve();run=a.reproduction.resolve();out=a.output.resolve()
    manifest=read(package/'release_manifest.json');audit=read(run/'reproduction_audit.json')
    assert audit['passed'] and audit['release_manifest_sha256']==sha(package/'release_manifest.json')
    assert len(audit['exact_arrays'])==12 and all(audit['exact_arrays'].values())
    assert all(v==0 for v in audit['maximum_differences'].values())
    assert audit['endpoint_contact_records_exact'] and audit['runtime_metrics_exact']
    assert audit['runtime']['physical_reset_count']==1 and audit['runtime']['success']
    assert audit['trajectory_sha256']==sha(run/'trajectory.npz')
    assert audit['golden_audit_sha256']==manifest['original_audit_sha256']
    assert audit['new_training_steps']==0 and audit['simulation_only']
    for relative,expected in manifest['files'].items():
        digest=expected['sha256'] if isinstance(expected,dict) else expected
        assert sha(package/relative)==digest,relative
    archive=out.with_suffix('.zip')
    assert not out.exists() and not archive.exists()
    out.mkdir(parents=True)
    shutil.copytree(package,out/'simulation')
    (out/'verification').mkdir()
    shutil.copy2(run/'reproduction_audit.json',out/'verification/reproduction_audit.json')
    # Preserve the exporter-era README/status as historical evidence. This
    # outer release describes the subsequently completed reproduction.
    (out/'README_先读.txt').write_text('''G1 v2 连续攀爬：已完成搬迁复现的仿真包

验证结果
在本机 Windows / Python 3.12 的另一个带空格目录中，隔离导入包内源码，
重新执行完整 909.28 秒 MuJoCo 动力学及冻结策略推理。12 组数组逐元素一致，
末段接触记录与运行指标一致，8 个周期实际累计爬升约 1.99853 米。
详见 verification/reproduction_audit.json。

使用方法（PowerShell，当前目录为解压后的外层目录）
  cd simulation
  py -3.12 -m venv .venv
  .\\.venv\\Scripts\\python.exe -m pip install -r requirements-runtime.txt
  .\\.venv\\Scripts\\python.exe -m deploy_mujoco.run_portable_v2_multicycle --verify-only
  .\\.venv\\Scripts\\python.exe -m deploy_mujoco.run_portable_v2_multicycle --output my_run

若需观察窗口，在最后一条命令增加 --viewer，且选择另一个不存在的输出目录。
窗口提前关闭不算完整验证。完成后查看 my_run/reproduction_audit.json 的 passed。
本包不连接真机。实际运行速度取决于电脑；909.28 秒是仿真时长。

验证边界
此处只验证同一台机器上的路径搬迁和现有依赖环境；尚未验证全新机器安装、
Linux 或不同版本依赖。每项依赖版本已固定。若安装失败，应保留安装日志，
不能直接解除版本固定后宣称与原验证一致。
固定动作参考、仿真真值杆位和动作校准仍存在；没有新增训练。
单个 T12 工况的可迁移复现不等于其他墙面或硬件泛化。

历史文件
simulation/README_zh.md 与 release_manifest.json 保留导出时的待验证描述，
不修改其原始哈希。本次完成情况由外层 verification 和 verified_release.json 记录。
simulation/evidence 中的旧绝对路径仅用于溯源，运行入口使用包内资源。
许可和模型来源保留在 simulation 中；未给混合快照添加统一的新许可证。
''',encoding='utf8')
    files={path.relative_to(out).as_posix():sha(path) for path in sorted(out.rglob('*')) if path.is_file()}
    write(out/'verified_release.json',dict(status='same_machine_relocated_full_physics_verified',files=files,
        original_release_manifest_sha256=sha(package/'release_manifest.json'),
        reproduction_audit_sha256=sha(run/'reproduction_audit.json'),
        fresh_install_verified=False,cross_platform_verified=False,new_training_steps=0,source_sha256=sha(__file__)))
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(out.parent))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        import hashlib
        for relative,digest in files.items():assert hashlib.sha256(z.read(out.name+'/'+relative)).hexdigest()==digest
    write(archive.with_suffix('.zip.verification.json'),dict(passed=True,zip_sha256=sha(archive),zip_bytes=archive.stat().st_size,
        archive_members=len(files)+1,all_packaged_files_exact=True,release_sha256=sha(out/'verified_release.json'),
        source_sha256=sha(__file__),scope='Archive CRC and hashes verified; full simulation reproduction is the separately bound evidence.'))
    print(json.dumps(dict(passed=True,archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive))),flush=True)


if __name__=='__main__':main()
