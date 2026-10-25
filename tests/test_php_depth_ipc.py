"""Offline camera process boundary: synthetic frames only, no USB/DDS/motors."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import yaml
from policy.php_parkour.real_depth import DepthSample, load_camera_config
from policy.php_parkour.depth_ipc import DepthReader, DepthWriter

ROOT=Path(__file__).resolve().parents[1]

class DepthIPCTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'depth.npz'
        self.cfg=load_camera_config(ROOT/'policy/php_parkour/config/real_depth.yaml')
        self.reader=DepthReader(self.cfg,self.path)
        self.writer=DepthWriter(self.path,self.cfg)

    def sample(self,seq=1,stamp=None):
        return DepthSample(np.full((58,87),.1,np.float32),time.monotonic() if stamp is None else stamp,seq,1.,'')

    def test_roundtrip_repoll_keeps_capture_age_and_shutdown_invalidates(self):
        sample=self.sample();self.writer.write(sample)
        np.testing.assert_array_equal(self.reader.poll().image,sample.image)
        self.assertEqual(self.reader.poll().captured_at,sample.captured_at)
        with patch('policy.php_parkour.depth_ipc.time.monotonic',return_value=sample.captured_at+.2):
            self.assertIsNone(self.reader.latest().image)
        self.writer.write(DepthSample(error='Camera stopped'))
        self.assertIn('Camera stopped',self.reader.poll().error)
        restarted=DepthWriter(self.path,self.cfg);restarted.write(self.sample())
        self.assertIsNotNone(self.reader.poll().image)

    def test_missing_malformed_mismatch_and_reordered_packets(self):
        self.assertIsNone(self.reader.poll().image)
        self.path.write_bytes(b'corrupt');self.assertIsNone(self.reader.poll().image)
        self.writer.write(self.sample(stamp=time.monotonic()+1));self.assertIsNone(self.reader.poll().image)
        self.writer.write(self.sample(5));self.assertIsNotNone(self.reader.poll().image)
        self.writer.write(self.sample(4));self.assertIsNone(self.reader.poll().image)
        other=copy.deepcopy(self.cfg);other['mount']['rpy_deg'][1]=37
        DepthWriter(self.path,other).write(self.sample(6))
        self.assertIn('mismatch',self.reader.poll().error)

    def test_separate_producer_process_and_reader(self):
        code='''import sys,time,numpy as np
from policy.php_parkour.depth_ipc import DepthWriter
from policy.php_parkour.real_depth import DepthSample,load_camera_config
DepthWriter(sys.argv[1],load_camera_config(sys.argv[2])).write(DepthSample(np.full((58,87),.2,np.float32),time.monotonic(),1,1.,''))
'''
        # Producer exits normally after a synthetic frame; no camera SDK is imported.
        # Slow Python process shutdown counts against the 150ms freshness threshold.
        proc=subprocess.Popen([sys.executable,'-c',code,str(self.path),str(ROOT/'policy/php_parkour/config/real_depth.yaml')],cwd=ROOT)
        try:
            deadline=time.monotonic()+15
            while not self.path.exists() and proc.poll() is None and time.monotonic()<deadline:time.sleep(.005)
            self.assertIsNotNone(self.reader.poll().image)
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()

    def test_camera_selection_check_is_read_only_and_independent_of_php(self):
        original=yaml.safe_load((ROOT/'config/deploy.yaml').read_text())
        for mode in ('gray','php_depth','invalid'):
            original['camera']={'mode':mode}
            original['php']['real']['enabled']=False
            config=Path(self.temp.name)/'deploy.yaml';config.write_text(yaml.safe_dump(original))
            result=subprocess.run([sys.executable,str(ROOT/'tools/capture/run_camera.py'),'--check'],
                env=dict(os.environ,LYD_DEPLOY_CONFIG=str(config)),capture_output=True,text=True)
            if mode=='invalid':self.assertNotEqual(result.returncode,0)
            else:self.assertEqual(result.stdout.strip(),mode);self.assertEqual(result.returncode,0)
