import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from tools.continue_vector_full_cycle_training import state_difference, VectorProgress
from training.full_cycle_residual_env import ROOT


class VectorContinuationTest(unittest.TestCase):
    def test_optimizer_state_comparison_detects_moment_change(self):
        layer = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(layer.parameters(), lr=3e-5)
        layer(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        original = optimizer.state_dict()
        duplicate = copy.deepcopy(original)
        self.assertEqual(state_difference(original, duplicate), 0.)
        first = next(iter(duplicate['state'].values()))
        first['exp_avg'].view(-1)[0] += .1
        self.assertGreater(state_difference(original, duplicate), 0.)
        self.assertNotEqual(state_difference({'x': torch.zeros(1)}, {}), 0.)
        self.assertNotEqual(state_difference(torch.zeros(1), torch.zeros(1, dtype=torch.float64)), 0.)
        self.assertNotEqual(state_difference(torch.tensor([float('nan')]), torch.tensor([float('nan')])), 0.)

    def test_vector_callback_counts_each_worker_and_logs_terminal_array(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs', prefix='vector_callback_test_') as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to((ROOT/'outputs').resolve()))
            callback = VectorProgress(directory, 20480)
            callback.num_timesteps = 20484
            callback.locals = {'infos': [
                {'case': 'nominal'},
                {'case': 'low', 'success': False, 'episode': {'r': np.float32(2.), 'l': 10}, 'terminal_observation': np.zeros(136)},
                {'case': 'low'}, {'case': 'high'}]}
            self.assertTrue(callback._on_step())
            self.assertEqual(callback.case_samples, {'nominal': 1, 'low': 2, 'high': 1})
            record = json.loads((Path(directory)/'training_episodes.jsonl').read_text())
            self.assertEqual(record['worker'], 1)
            self.assertEqual(record['global_step'], 20484)
            self.assertFalse(record['success'])
            self.assertEqual(len(record['terminal_observation']), 136)


if __name__ == '__main__':
    unittest.main()
