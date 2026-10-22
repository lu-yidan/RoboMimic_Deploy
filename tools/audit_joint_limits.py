"""Additional native position-range audit, separate from legacy cycle checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from training.wall_support_env import WallSupportEnv


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text());e=WallSupportEnv(manifest['bundle'])
    z=dict(np.load(a.run/'trajectory.npz'));lo=e.model.jnt_range[e.joints,0];hi=e.model.jnt_range[e.joints,1]
    margin=np.minimum(z['qpos'][:,e.qadr]-lo,hi-z['qpos'][:,e.qadr]);margin[:,~e.model.jnt_limited[e.joints].astype(bool)]=np.inf
    i,j=np.unravel_index(np.argmin(margin),margin.shape);tolerance=1e-7
    violating=np.any(margin < -tolerance,axis=1)
    result=dict(passed=bool(not np.any(violating)),tolerance_rad=tolerance,
        minimum_margin_rad=float(margin[i,j]),worst_joint=e.model.joint(e.joints[j]).name,
        worst_time_s=float(z['time_s'][i]),worst_position_rad=float(z['qpos'][i,e.qadr[j]]),native_range_rad=[float(lo[j]),float(hi[j])],
        violating_samples=int(sum(violating)),sample_count=len(margin),
        trajectory_sha256=hashlib.sha256((a.run/'trajectory.npz').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Saved 50 Hz native joint position ranges, tolerance only for floating-point roundoff. This is additional to legacy acceptance; no between-sample bound claim.')
    (a.run/'joint_limit_audit.json').write_text(json.dumps(result,indent=2));(a.run/'joint_limit_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2));e.close()
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
