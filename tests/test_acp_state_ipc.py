"""ACP localization/terrain IPC tests; no hardware, DDS, or camera access."""
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from policy.acp.state_ipc import StateReader, StateWriter
from policy.acp.terrain import flat_grid


class ACPStateIPCTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / 'state.npz'
        self.config = {'max_age_s': .15, 'max_sync_skew_s': .04, 'real': {'frame_id': 'map'}}
        self.reader, self.writer = StateReader(self.config, self.path), StateWriter(self.path)

    def write(self, sequence=1, stamp=None, **kwargs):
        stamp = time.monotonic() if stamp is None else stamp
        values = dict(pelvis_pos_w=[.2, -.1, .8], pelvis_quat_w=[1., 0., 0., 0.],
                      captured_at=stamp, sequence=sequence, ground_height=.1)
        values.update(kwargs)
        self.writer.write(**values)
        return stamp

    def test_plane_and_grid_preserve_capture_timestamps(self):
        stamp = self.write()
        sample = self.reader.poll()
        self.assertFalse(sample.error)
        np.testing.assert_allclose(sample.terrain[:, 2], .2, atol=1e-6)
        self.assertEqual(self.reader.poll().captured_at, stamp)
        self.assertFalse(sample.terrain.flags.writeable)
        self.write(2, terrain=flat_grid(.8, .1), ground_height=None)
        self.assertFalse(self.reader.poll().error)
        self.reader.close()
        self.assertIn('stopped', self.reader.latest().error)

    def test_no_default_pose_or_plane_and_wrong_frame_are_invalid(self):
        self.assertTrue(self.reader.poll().error)
        with self.assertRaises(ValueError):
            self.write(ground_height=None)
        self.write(pelvis_pos_w=[0., 0., float('nan')])
        self.assertTrue(self.reader.poll().error)
        self.writer = StateWriter(self.path, frame_id='odom')
        self.write(2)
        self.assertIn('frame mismatch', self.reader.poll().error)

    def test_stale_future_unsynchronized_and_repeat_packets(self):
        self.write(stamp=time.monotonic()-.2)
        self.assertTrue(self.reader.poll().error)
        self.write(stamp=time.monotonic()+1)
        self.assertTrue(self.reader.poll().error)
        self.write(pose_captured_at=time.monotonic()-.1)
        self.assertTrue(self.reader.poll().error)
        stamp = self.write(4)
        self.assertFalse(self.reader.poll().error)
        with patch('policy.acp.state_ipc.time.monotonic', return_value=stamp+.2):
            self.assertIn('stale', self.reader.latest().error)
        self.write(4, stamp)
        self.assertIn('Repeated', self.reader.poll().error)
        self.write(5, stamp)
        self.assertIn('Repeated', self.reader.poll().error)
        self.write(3)
        self.assertIn('Repeated', self.reader.poll().error)
        self.write(6)
        self.assertFalse(self.reader.poll().error)

    def test_invalid_grid_quaternion_and_corrupt_file(self):
        grid = flat_grid(.8)
        grid[[0, 1]] = grid[[1, 0]]
        self.write(terrain=grid, ground_height=None)
        self.assertIn('grid order', self.reader.poll().error)
        self.write(2, pelvis_quat_w=[0., 0., 0., 0.])
        self.assertIn('pelvis', self.reader.poll().error)
        self.path.write_bytes(b'incomplete producer output')
        self.assertTrue(self.reader.poll().error)


if __name__ == '__main__':
    unittest.main()
