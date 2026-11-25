"""Pin and patch official sources in a dedicated, owned colcon workspace."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def apply_patch_stack(dest, patches):
    """Upgrade owned patch stacks without resetting or discarding local edits.

    Later patches may overlap earlier contexts. Unwind recognized patches in
    reverse order, then apply the entire stack. Restore affected bytes on error.
    """
    snapshots = {}
    for patch in patches:
        for line in run('git', '-C', str(dest), 'apply', '--numstat', str(patch)).splitlines():
            relative = Path(line.split('\t', 2)[2])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Patch path escapes source directory')
            path = dest / relative
            snapshots.setdefault(path, path.read_bytes() if path.exists() else None)
    try:
        for patch in reversed(patches):
            check = subprocess.run(['git', '-C', str(dest), 'apply', '--reverse', '--check', str(patch)],
                                   capture_output=True)
            if check.returncode == 0:
                run('git', '-C', str(dest), 'apply', '--reverse', str(patch))
        for patch in patches:
            run('git', '-C', str(dest), 'apply', '--check', str(patch))
            run('git', '-C', str(dest), 'apply', str(patch))
    except BaseException:
        for path, content in snapshots.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        raise


def main():
    ws = Path(sys.argv[1]).expanduser().resolve()
    marker = ws / '.acp-dlio-workspace'
    if not marker.exists() and any((ws / p).exists() for p in ('build', 'install', 'src/direct_lidar_inertial_odometry', 'src/livox_ros_driver2')):
        raise SystemExit('Refusing to modify an existing workspace without ACP ownership marker')
    ws.mkdir(parents=True, exist_ok=True)
    marker.touch()
    locks = json.loads((HERE / 'sources.json').read_text())
    result = {}
    for key, name in [('livox', 'livox_ros_driver2'), ('dlio', 'direct_lidar_inertial_odometry')]:
        spec = locks[key]
        dest = ws / 'src' / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            run('git', 'init', str(dest))
            run('git', '-C', str(dest), 'remote', 'add', 'origin', spec['url'])
            run('git', '-C', str(dest), 'fetch', '--depth', '1', 'origin', spec['commit'])
            run('git', '-C', str(dest), 'checkout', '--detach', 'FETCH_HEAD')
        if run('git', '-C', str(dest), 'rev-parse', 'HEAD') != spec['commit']:
            raise SystemExit('Unexpected upstream HEAD: ' + str(dest))
        patch = HERE / 'patches' / (key + '.patch')
        patches = [p for p in (patch, HERE / 'patches' / (key + '-runtime.patch'), HERE / 'patches' / (key + '-state.patch')) if p.exists()]
        apply_patch_stack(dest, patches)
        if key == 'livox':
            shutil.copyfile(dest / 'package_ROS2.xml', dest / 'package.xml')
        result[key] = dict(spec, patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest() if patch.exists() else None,
                           patches=[dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in patches])
    (ws / 'acp-sources.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
