"""Right-top camera process selected by camera.mode. No robot control or bridge."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import sys
import threading

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common.deployment_config import load_deployment_config, load_php_real_config


def camera_mode():
    mode=load_deployment_config().get('camera',{}).get('mode','gray')
    if mode not in ('gray','php_depth'):
        raise ValueError('camera.mode must be gray or php_depth')
    return mode


def run_depth():
    # Independent of policy enablement: camera-only preview is supported.
    from policy.php_parkour.real_depth import RealSenseDepth, load_camera_config
    from policy.php_parkour.depth_ipc import DepthWriter, depth_path
    from common.php_depth_preview import DepthPreviewPublisher, preview_path
    import torch
    torch.set_num_threads(1)
    config=load_camera_config(load_php_real_config()['camera_config'])
    path=depth_path();path.parent.mkdir(parents=True,exist_ok=True)
    # Only one publisher per IPC path; keep the lock inode in place on exit.
    with path.with_suffix('.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('PHP camera process is already running for '+str(path))
        source=RealSenseDepth(config)
        writer=DepthWriter(path,config)
        preview=DepthPreviewPublisher(source,preview_path())
        stop=threading.Event()
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
        previous=None
        try:
            writer.write(source.latest())
            source.start();preview.start()
            print('[PHP camera] Independent capture; Dashboard http://<robot-ip>:8091/; Ctrl-C stops camera only',flush=True)
            while not stop.is_set():
                sample=source.latest()
                key=(sample.sequence,sample.captured_at,sample.error)
                if key!=previous:
                    writer.write(sample)
                    if previous is None or key[2]!=previous[2]:
                        print('[PHP camera] '+(sample.error or 'depth ready'),flush=True)
                    previous=key
                stop.wait(.005)
        finally:
            source.close()
            try:writer.write(source.latest())
            finally:preview.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Print selection without opening USB or DDS')
    args=parser.parse_args()
    mode=camera_mode()
    if args.check:
        print(mode);return
    if mode=='gray':
        print('[Camera] Gray MJPEG http://<robot-ip>:8080/',flush=True)
        os.chdir(ROOT)
        os.execv('/bin/bash',['bash',str(ROOT/'onboard/perception/camera/run_gray.sh'),'--show'])
    run_depth()


if __name__=='__main__':main()
