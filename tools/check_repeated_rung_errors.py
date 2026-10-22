"""Focused geometry/continuity checks for isolated two-cycle scene errors."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.repeated_rung_errors import apply_rung_offsets
from training.climb_robustness import apply_rung_offsets as old

def main():
    x=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');y=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');x.reset();y.reset()
    old(x,{'rung_01':[0.,0.,.005]});apply_rung_offsets(y,{'rung_01':[0.,0.,.005]})
    equal=all(np.array_equal(getattr(x.model,key),getattr(y.model,key)) for key in ['geom_pos','site_pos','bvh_aabb','bvh_child','bvh_nodeid']);x.close();y.close()
    e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset();e.data.qvel[:]=.001
    initial=e.model.geom_pos.copy();q=e.data.qpos.copy();v=e.data.qvel.copy();index=e.model.jnt_qposadr.copy();rejected=[]
    for offsets in [{'rung_01':[0,0,.005],'rung_07':[0,0,.005]},{'rung_01':[0,0,.005],'rung_02':[0,0,float('nan')]}]:
        try:apply_rung_offsets(e,offsets)
        except ValueError:rejected.append(True)
        else:rejected.append(False)
        assert np.array_equal(initial,e.model.geom_pos)
    variant=apply_rung_offsets(e,{'rung_01':[0.,0.,.005],'rung_02':[0.,0.,.005]})
    checks=dict(single_row_exact_compatibility=equal,invalid_requests_atomic=all(rejected),qpos_unchanged=np.array_equal(q,e.data.qpos),qvel_unchanged=np.array_equal(v,e.data.qvel),joint_index_unchanged=np.array_equal(index,e.model.jnt_qposadr),recompiled_topology_checked='topology verified identical' in variant['static_bvh'])
    result=dict(passed=bool(all(checks.values())),checks={k:bool(v) for k,v in checks.items()},geometry_sha256=variant['geometry_sha256']);e.close();print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
