import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.train_full_cycle_residual import Progress
from tools.train_strict_full_cycle_residual import Progress as StrictProgress
from training.full_cycle_residual_env import ROOT


class EpisodeLoggingTest(unittest.TestCase):
    def test_sb3_terminal_observation_round_trip(self):
        for callback_class in (Progress, StrictProgress):
            with tempfile.TemporaryDirectory(dir=ROOT/'outputs', prefix='episode_log_test_') as directory:
                self.assertTrue(Path(directory).resolve().is_relative_to((ROOT/'outputs').resolve()))
                callback = callback_class(Path(directory))
                callback.num_timesteps = 5536
                callback.locals = {'infos': [{'episode': {'r': np.float64(120.), 'l': np.int64(5536)},
                    'success': np.bool_(True), 'terminal_observation': np.arange(136, dtype=np.float32)}]}
                self.assertTrue(callback._on_step())
                record = json.loads((Path(directory)/'training_episodes.jsonl').read_text())
                self.assertEqual(record['terminal_observation'], list(range(136)))
                self.assertTrue(record['success'])
                self.assertEqual(record['episode']['l'], 5536)


if __name__ == '__main__':
    unittest.main()
