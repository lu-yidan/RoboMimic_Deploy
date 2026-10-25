"""Ensure PHP launch selection preserves paths and forwards arguments without a shell."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from deploy_php.deploy_php import command


class PHPLauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='php models ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('run_php_sim.sh', 'run_php_inference.sh', 'depth_backbone.onnx', 'student.onnx',
                     'thirdparty/holosoma/scripts/source_mujoco_setup.sh',
                     'thirdparty/holosoma/scripts/source_inference_setup.sh'):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        patcher = patch('deploy_php.deploy_php.settings', return_value=({'scene': 'steps'}, self.root, self.root))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_scene_override_and_raw_argument(self):
        cmd, env, root = command('sim', 'flat', ['--label', 'a b;$HOME'])
        self.assertIn('terrain:terrain-locomotion-plane', cmd)
        self.assertEqual(cmd[-1], 'a b;$HOME')
        self.assertEqual(env['CONDA_ENV_NAME'], 'hsmujoco')
        self.assertEqual(root, self.root)

    def test_policy_models_and_missing_file(self):
        _, env, _ = command('policy')
        self.assertEqual(env['STUDENT'], str(self.root / 'student.onnx'))
        self.assertEqual(env['CONDA_ENV_NAME'], 'hsinference')
        (self.root / 'student.onnx').unlink()
        with self.assertRaises(FileNotFoundError):
            command('policy')

    def test_invalid_scene_and_missing_checkout(self):
        with self.assertRaises(ValueError):
            command('sim', 'unknown')
        (self.root / 'thirdparty/holosoma/scripts/source_mujoco_setup.sh').unlink()
        with self.assertRaises(FileNotFoundError):
            command('sim')
