"""Best-effort onboard preview IPC. Never used as a control input."""
import base64
import io
import json
import os
from pathlib import Path
import threading
import time


def preview_path():
    from common.deployment_config import load_deployment_config
    cfg = load_deployment_config().get('php', {}).get('real', {})
    return Path(cfg.get('dashboard_file', f'/dev/shm/robomimic-{os.getuid()}/php-depth.json')).expanduser()


def read_preview(path):
    try:
        data = json.loads(Path(path).read_text())
        age = time.monotonic() - float(data['captured_at'])
        heartbeat_age = time.monotonic() - float(data['updated_at'])
        fresh = 0 <= age <= min(.15, float(data['max_age_s'])) and 0 <= heartbeat_age <= .5
        data['age_ms'] = max(0, age * 1000) if data['captured_at'] else None
        data['valid'] = bool(data.get('image')) and fresh and not data.get('error')
        if not data['valid']:
            data['image'] = None
            data['error'] = data.get('error') or 'Depth stream stalled/stale'
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return {'valid': False, 'image': None, 'age_ms': None,
                'error': 'PHP depth unavailable (disabled or producer not running)'}


class DepthPreviewPublisher:
    """10 Hz worker: snapshot only, no blocking I/O or encoding in control loop."""
    def __init__(self, source, path):
        self.source, self.path = source, Path(path)
        self.stop = threading.Event()
        self.thread = None

    def publish_once(self):
        import numpy as np
        from PIL import Image
        sample = self.source.latest()
        encoded = None
        if sample.image is not None:
            near = 1 - np.clip(sample.image + .5, 0, 1)
            rgb = np.stack((255*near, 60+160*near, 200*(1-near)), -1).astype(np.uint8)
            out = io.BytesIO()
            Image.fromarray(rgb).save(out, format='PNG')
            encoded = base64.b64encode(out.getvalue()).decode('ascii')
        data = dict(image=encoded, captured_at=sample.captured_at,
                    updated_at=time.monotonic(), sequence=sample.sequence,
                    coverage=sample.valid_fraction, error=sample.error,
                    max_age_s=self.source.config['max_age_s'])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + f'.{os.getpid()}.tmp')
        try:
            temporary.write_text(json.dumps(data, allow_nan=False))
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _run(self):
        warned = False
        while not self.stop.is_set():
            try:
                self.publish_once()
                warned = False
            except Exception as error:
                if not warned:
                    print('[PHP preview] '+str(error), flush=True)
                warned = True
            self.stop.wait(.1)

    def start(self):
        self.thread = threading.Thread(target=self._run, name='php-preview', daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.)
        # Source is closed first; old data also expires by its capture timestamp.
        try:
            self.publish_once()
        except Exception:
            pass
