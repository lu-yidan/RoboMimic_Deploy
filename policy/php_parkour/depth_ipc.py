"""Local atomic depth transport. Capture timestamps are never refreshed by readers."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np
from policy.php_parkour.real_depth import DepthSample


def camera_signature(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def depth_path():
    from common.deployment_config import load_deployment_config
    real=load_deployment_config().get('php',{}).get('real',{})
    return Path(real.get('depth_file',f'/dev/shm/robomimic-{os.getuid()}/php-depth.npz')).expanduser()


class DepthWriter:
    def __init__(self, path, config):
        self.path=Path(path)
        self.signature=camera_signature(config)
        self.session=uuid.uuid4().hex
        self.boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()

    def write(self, sample):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_name(self.path.name+f'.{os.getpid()}.tmp')
        meta=dict(version=1,boot=self.boot,session=self.session,signature=self.signature,
                  captured_at=sample.captured_at,sequence=sample.sequence,
                  coverage=sample.valid_fraction,error=sample.error)
        try:
            with temp.open('wb') as out:
                np.savez(out,meta=json.dumps(meta,allow_nan=False),
                         image=sample.image if sample.image is not None else np.empty((0,),np.float32))
            temp.replace(self.path)
        finally:
            temp.unlink(missing_ok=True)


class DepthReader:
    """A reader worker isolates filesystem operations from the 50 Hz control loop."""
    def __init__(self, config, path=None):
        self.config=config
        self.path=Path(path) if path is not None else depth_path()
        self.signature=camera_signature(config)
        self.boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.sample=DepthSample(error='Camera process not started')
        self.lock=threading.Lock()
        self.stop=threading.Event()
        self.thread=None
        self.previous=None

    def poll(self):
        try:
            if self.path.stat().st_size > 65536:
                raise ValueError('Oversized depth packet')
            with np.load(self.path,allow_pickle=False) as data:
                meta=json.loads(str(data['meta']))
                image=data['image'].copy()
            if meta['version']!=1 or meta['boot']!=self.boot or meta['signature']!=self.signature:
                raise ValueError('Depth protocol/boot/camera configuration mismatch')
            if meta['error']:
                raise ValueError(meta['error'])
            stamp=float(meta['captured_at']);seq=int(meta['sequence'])
            coverage=float(meta['coverage'])
            if not np.isfinite(stamp) or not 0 <= time.monotonic()-stamp <= self.config['max_age_s']:
                raise ValueError('Depth stream stalled/stale')
            if image.shape!=(58,87) or image.dtype!=np.float32 or not np.isfinite(image).all():
                raise ValueError('Invalid depth payload')
            if not self.config['min_valid_fraction'] <= coverage <= 1:
                raise ValueError('Invalid depth coverage')
            current=(str(meta['session']),seq,stamp)
            if self.previous is not None and current[0]==self.previous[0]:
                if seq<self.previous[1] or stamp<self.previous[2] or (seq==self.previous[1] and stamp!=self.previous[2]):
                    raise ValueError('Reordered or re-stamped depth packet')
            self.previous=current
            image.setflags(write=False)
            sample=DepthSample(image,stamp,seq,coverage,'')
        except Exception as error:
            sample=DepthSample(error='Camera IPC: '+str(error))
        with self.lock:
            self.sample=sample
        return sample

    def latest(self):
        with self.lock:
            sample=self.sample
        if sample.image is not None and not 0 <= time.monotonic()-sample.captured_at <= self.config['max_age_s']:
            return DepthSample(error='Depth stream stalled/stale')
        return sample

    def _run(self):
        while not self.stop.is_set():
            self.poll()
            self.stop.wait(.01)

    def start(self):
        self.thread=threading.Thread(target=self._run,name='php-depth-reader',daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:self.thread.join(timeout=1.)
        with self.lock:self.sample=DepthSample(error='Depth reader stopped')
