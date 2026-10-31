"""Static verification of the frozen hook grid; no dynamics are stepped."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.physical_v2_cycle_env import PhysicalV2CycleEnv
from training.compiled_rung_geometry import CompiledPhysicalV2CycleEnv
from training.cycle_relative_bar_observation import CycleRelativeBarObservation
from tools.validate_swing_reference import read,write,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--baseline-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scene-compiler',choices=('legacy','complete'),default='legacy')
    args=parser.parse_args();protocol=read(args.protocol)
    run=args.baseline_run.resolve();manifest=read(run/'manifest.json')
    audit=read(run/'v2_multicycle_audit.json')
    assert audit['passed'] and audit['trajectory_sha256']==sha(run/'trajectory.npz')
    assert manifest['case']=='training_12' and manifest['cycles']==2
    parent=read(Path(manifest['parent_experiment'])/'manifest.json')
    with np.load(run/'trajectory.npz') as z:
        fixed_poses={group:z['qpos'][int(np.argmin(abs(z['time_s']-when)))].copy()
                     for group,when in [('right_hook',127.66),('left_hook',140.66)]}
    environment=PhysicalV2CycleEnv if args.scene_compiler=='legacy' else CompiledPhysicalV2CycleEnv
    base=environment(plan=manifest['plan_path'],cases=protocol['cases'],
        normal_bias_y_deg=parent['normal_bias_y_deg'],memory_enabled=parent['force_history_memory'])
    observer=CycleRelativeBarObservation(base,mode='oracle');e=base.env
    rows=[]
    try:
        for case,offsets in protocol['cases'].items():
            observer.reset(options={'case':case})
            expected=e.model.geom_pos.copy();expected[:]=base.nominal_geometry[0]
            moved=[]
            for i in range(e.model.ngeom):
                name=e.model.geom(i).name or ''
                for rod,delta in offsets.items():
                    if name==rod or name.startswith(rod+'_'):
                        expected[i]+=delta;moved.append(name)
            if args.scene_compiler=='legacy':assert np.array_equal(expected,e.model.geom_pos)
            else:np.testing.assert_allclose(expected,e.model.geom_pos,rtol=0,atol=1e-12)
            geometry_roundoff=float(abs(expected-e.model.geom_pos).max(initial=0))
            for cycle,time in [(0,0.),(1,113.6601)]:
                e.data.time=time
                observation=observer.observation(base.observation())
                requested=np.array([offsets.get(rod,[0.,0.,0.]) for rod in observer.role_rows[cycle]]).reshape(-1)/.02
                assert np.max(abs(observation[-12:]-requested))<1e-7
            distances={}
            target=e.model.geom('rung_06').id
            for group,pose in fixed_poses.items():
                e.data.qpos[:]=pose;mujoco.mj_forward(e.model,e.data)
                candidates=[i for i in range(e.model.ngeom)
                    if e.contact_groups.get(int(e.model.geom_bodyid[i]))==group
                    and (e.model.geom_contype[i] or e.model.geom_conaffinity[i])]
                values=[(float(mujoco.mj_geomDistance(e.model,e.data,i,target,.05,np.zeros(6))),i) for i in candidates]
                distance,geom=min(values)
                distances[group]=dict(distance_mm=distance*1000,nearest_geom=e.model.geom(geom).name)
            rows.append(dict(case=case,passed=True,offsets=offsets,moved_geoms=moved,geometry_roundoff_m=geometry_roundoff,
                geometry_sha256=base.variant['geometry_sha256'],fixed_baseline_pose_gaps=distances))
    finally:observer.close()
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(passed=True,cases=len(rows),rows=rows,physics_steps=0,scene_compiler=args.scene_compiler,
        protocol_sha256=sha(args.protocol),baseline_audit_sha256=sha(run/'v2_multicycle_audit.json'),
        baseline_trajectory_sha256=sha(run/'trajectory.npz'),source_sha256=sha(__file__),
        scope='Geometry-only preflight: named rods and mounts move together, unrelated geometry remains nominal, and oracle features reflect the declared translations in both cycle roles. Gaps are measured at original baseline poses in the changed world, not trajectories produced in those geometries. No contact-force, timing or dynamic-success conclusion.')
    write(args.output/'summary.json',report)
    (args.output/'preflight_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(passed=True,cases=len(rows),physics_steps=0,
        left_gap_mm=[min(r['fixed_baseline_pose_gaps']['left_hook']['distance_mm'] for r in rows),
                     max(r['fixed_baseline_pose_gaps']['left_hook']['distance_mm'] for r in rows)])),flush=True)


if __name__=='__main__':main()
