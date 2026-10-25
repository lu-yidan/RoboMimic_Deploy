"""Offline preview IPC + actual Dashboard HTTP; no SDK or DDS participant."""
import base64
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import urlopen
from socketserver import ThreadingTCPServer

import numpy as np
from PIL import Image
from common.php_depth_preview import DepthPreviewPublisher, read_preview
from policy.php_parkour.real_depth import DepthBuffer, load_camera_config, Intrinsics
from onboard.perception.debug.sensor_dashboard import SensorStore, HTML, _make_handler

ROOT = Path(__file__).resolve().parents[1]

class DashboardDepthTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)/'depth.json'
        cfg=load_camera_config(ROOT/'policy/php_parkour/config/real_depth.yaml')
        self.source=DepthBuffer(cfg)
        self.publisher=DepthPreviewPublisher(self.source,self.path)

    def frame(self):
        f=53/np.tan(np.deg2rad(89.5/2))
        self.source.submit(np.full((60,106),1.5,np.float32),Intrinsics(106,60,f,f,53,30),time.monotonic(),1)
        self.publisher.publish_once()

    def test_image_matches_model_input_and_expires(self):
        self.frame()
        result=read_preview(self.path)
        self.assertTrue(result['valid'])
        image=Image.open(io.BytesIO(base64.b64decode(result['image'])))
        self.assertEqual(image.size,(87,58))
        near=1-np.clip(self.source.latest().image+.5,0,1)
        expected=np.stack((255*near,60+160*near,200*(1-near)),-1).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(image),expected)
        with patch('common.php_depth_preview.time.monotonic',return_value=time.monotonic()+1):
            stale=read_preview(self.path)
        self.assertFalse(stale['valid']);self.assertIsNone(stale['image'])
        self.source.invalidate('USB disconnected');self.publisher.publish_once()
        self.assertEqual(read_preview(self.path)['error'],'USB disconnected')

    def test_missing_corrupt_preview_and_failed_writer_do_not_affect_control(self):
        self.assertFalse(read_preview(self.path)['valid'])
        self.path.write_text('{broken')
        self.assertFalse(read_preview(self.path)['valid'])
        self.frame()
        with patch.object(Path,'mkdir',side_effect=OSError('read-only')):
            with self.assertRaises(OSError):self.publisher.publish_once()
        self.assertIsNotNone(self.source.latest().image)

    def test_actual_http_serves_preview_without_dds(self):
        with patch('cyclonedds.domain.DomainParticipant.__init__',side_effect=AssertionError('DDS opened')):
            self.frame()
            store=SensorStore(400,400,php_depth_path=self.path)
            html=HTML.replace('__RANGE_M__','4').encode()
            with ThreadingTCPServer(('127.0.0.1',0),_make_handler(store,html,.05)) as server:
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                try:
                    url='http://127.0.0.1:'+str(server.server_address[1])
                    page=urlopen(url).read().decode()
                    self.assertIn('php-depth-image',page)
                    self.assertIn('phpDeadline',page)
                    data=json.load(urlopen(url+'/state'))
                    self.assertTrue(data['phpDepth']['valid'])
                    self.source.invalidate('USB disconnected');self.publisher.publish_once()
                    data=json.load(urlopen(url+'/state'))
                    self.assertFalse(data['phpDepth']['valid'])
                    self.assertIsNone(data['phpDepth']['image'])
                finally:
                    server.shutdown();thread.join()
