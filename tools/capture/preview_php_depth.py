"""Read-only RealSense capture/preview. Imports no bridge, DDS or robot SDK.

This tool opens USB only when explicitly invoked. It never runs a controller.
"""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from policy.php_parkour.real_depth import load_camera_config, RealSenseDepth, DepthProjector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera-config',type=Path,default=ROOT/'policy/php_parkour/config/real_depth.yaml')
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--out',type=Path,default=Path('/tmp/php-depth.npz'))
    args=p.parse_args()
    if not 0 < args.seconds <= 600:
        p.error('seconds must be in (0,600]')
    cfg=load_camera_config(args.camera_config)
    import torch
    torch.set_num_threads(1)
    projector=DepthProjector(cfg)
    source=RealSenseDepth(cfg);source.start()
    saved=False
    try:
        end=time.monotonic()+args.seconds
        last=-1
        while time.monotonic()<end:
            sample=source.latest()
            if sample.image is None:
                print('[PHP camera] '+sample.error,flush=True)
            elif sample.sequence!=last:
                last=sample.sequence
                depth,k,stamp=source.recording()
                # Derive preview from the SAME recording to avoid mixed-frame evidence.
                normalized,coverage=projector.project(depth,k)
                age=time.monotonic()-stamp
                if age <= cfg['max_age_s']:
                    np.savez_compressed(args.out,depth_m=depth,
                        intrinsics=[k.width,k.height,k.fx,k.fy,k.cx,k.cy],capture_age_s=age,
                        position_m=cfg['mount']['position_m'],rpy_deg=cfg['mount']['rpy_deg'])
                    near=1-np.clip(normalized+.5,0,1)
                    rgb=np.stack((255*near,60+160*near,200*(1-near)),-1).astype(np.uint8)
                    Image.fromarray(rgb).resize((348,232),Image.Resampling.NEAREST).save(args.out.with_suffix('.png'))
                    print('frame={} coverage={:.1%} age={:.1f}ms'.format(last,coverage,age*1000),flush=True)
                    saved=True
            time.sleep(.1)
    finally:
        source.close()
    if not saved:
        raise SystemExit('No accepted depth frame; no policy was run.')
    print('Saved raw depth/intrinsics and PNG:',args.out)


if __name__=='__main__':
    main()
