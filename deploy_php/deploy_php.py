#!/usr/bin/env python3
"""Launch the external PHP/Holosoma depth-policy deployment from this repository."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.deployment_config import load_deployment_config


def settings():
    cfg = load_deployment_config().get('php', {})
    def path(key, default):
        value = Path(cfg.get(key, default)).expanduser()
        return value if value.is_absolute() else ROOT / value
    return cfg, path('root', '~/workspace/php_parkour'), path('model_dir', '~/.cache/php-parkour/student')


def command(mode, scene=None, extra=()):
    cfg, root, models = settings()
    env = os.environ.copy()
    # Do not accidentally pick another checkout or an inherited conda environment.
    env['HOLOSOMA_ROOT'] = str(root / 'thirdparty/holosoma')
    if mode == 'sim':
        script = root / 'run_php_sim.sh'
        scene = scene or cfg.get('scene', 'steps')
        terrains = {'steps': 'terrain:terrain-load-step', 'flat': 'terrain:terrain-locomotion-plane'}
        if scene not in terrains:
            raise ValueError('php.scene must be steps or flat')
        args = ['bash', str(script), terrains[scene]]
        env['CONDA_ENV_NAME'] = 'hsmujoco'
    else:
        script = root / 'run_php_inference.sh'
        for name in ('depth_backbone.onnx', 'student.onnx'):
            if not (models / name).is_file():
                raise FileNotFoundError('Missing model {}; run: python scripts/download_assets.py student in {}'.format(models / name, root))
        env.update(BACKBONE=str(models / 'depth_backbone.onnx'), STUDENT=str(models / 'student.onnx'), CONDA_ENV_NAME='hsinference')
        args = ['bash', str(script)]
    if not script.is_file():
        raise FileNotFoundError('Missing PHP launcher: {}'.format(script))
    setup = root / 'thirdparty/holosoma/scripts' / ('source_mujoco_setup.sh' if mode == 'sim' else 'source_inference_setup.sh')
    if not setup.is_file():
        raise FileNotFoundError('Initialize PHP submodule: git -C {} submodule update --init thirdparty/holosoma'.format(root))
    return args + list(extra), env, root


def check():
    """Offline model contract and inference check; never opens DDS or a camera."""
    import hashlib
    import numpy as np
    import onnxruntime as ort
    _, root, models = settings()
    command('sim')
    command('policy')
    # Report hashes for custom exports too, without claiming they are release weights.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sessions = [ort.InferenceSession(str(models / name), options, providers=['CPUExecutionProvider'])
                for name in ('depth_backbone.onnx', 'student.onnx')]
    backbone, student = sessions
    if backbone.get_inputs()[0].shape != [1, 58, 87] or student.get_inputs()[0].shape != [1, 140]:
        raise ValueError('Expected released PHP contract: depth [1,58,87], obs [1,140]')
    latent = backbone.run(None, {backbone.get_inputs()[0].name: np.zeros((1, 58, 87), np.float32)})[0]
    if latent.shape != (1, 32):
        raise ValueError('Expected 32 depth features')
    obs = np.zeros((1, 140), np.float32)
    obs[0, 93] = 1  # stand command, following the 93-dimensional proprioception
    obs[:, -32:] = latent
    feed = {'obs': obs, 'time_step': np.zeros((1, 1), np.float32)}
    actions = student.run(None, feed)[0]
    if actions.shape != (1, 29) or not np.isfinite(actions).all():
        raise ValueError('Invalid student actions')
    print(json.dumps({'result': 'offline inference passed', 'depth': [1, 58, 87], 'obs': list(obs.shape),
                      'actions': list(actions.shape), 'models': {
                          p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (models / 'depth_backbone.onnx', models / 'student.onnx')}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('sim', 'policy', 'check'))
    parser.add_argument('--scene', choices=('steps', 'flat'))
    parser.add_argument('--dry-run', action='store_true')
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    args = parser.parse_args(argv[:split])
    extra = argv[split + 1:]
    try:
        if args.mode == 'check':
            if args.dry_run or args.scene or extra:
                parser.error('check takes no launch options')
            check()
            return
        if args.scene and args.mode != 'sim':
            parser.error('--scene only applies to sim')
        cmd, env, root = command(args.mode, args.scene, extra)
        if args.dry_run:
            print('cwd: {}'.format(root))
            for key in ('HOLOSOMA_ROOT', 'CONDA_ENV_NAME', 'BACKBONE', 'STUDENT'):
                if key in env:
                    print('{}={}'.format(key, shlex.quote(env[key])))
            print(' '.join(shlex.quote(arg) for arg in cmd))
        else:
            os.chdir(str(root))
            os.execvpe(cmd[0], cmd, env)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, 'PHP deployment error: {}\n'.format(exc))


if __name__ == '__main__':
    main()
