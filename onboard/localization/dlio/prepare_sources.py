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
        if patch.exists():
            check = subprocess.run(['git', '-C', str(dest), 'apply', '--check', str(patch)], capture_output=True)
            if check.returncode == 0:
                run('git', '-C', str(dest), 'apply', str(patch))
            else:
                run('git', '-C', str(dest), 'apply', '--reverse', '--check', str(patch))
        if key == 'livox':
            shutil.copyfile(dest / 'package_ROS2.xml', dest / 'package.xml')
        result[key] = dict(spec, patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest() if patch.exists() else None)
    (ws / 'acp-sources.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
