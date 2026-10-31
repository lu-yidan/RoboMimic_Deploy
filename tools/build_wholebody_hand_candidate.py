"""Build a bounded whole-body target candidate and audit the sampled path."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
from training.compiled_rung_geometry import CompiledPhysicalV2CycleEnv
from training.wholebody_hand_reference import refine_wholebody_hand_targets
from training.densify_reference import densify_after
from tools.plan_clearance_cycle import audit_plan
from tools.probe_hand_reference_path import read, write, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--continuous', action='store_true', help='Warm-start successive solves on a maximum 20 ms knot spacing')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path('outputs/v2_swing_two_reference_20261029_v1/plan.npz')
    assert sha(source) == read(source.parent/'manifest.json')['plan_sha256']
    windows = {'lh': [136.66, 138.66], 'rh': [123.66, 125.66]}
    offset = [.002, 0., .002]
    sources = ('training/wholebody_hand_reference.py', 'tools/build_wholebody_hand_candidate.py',
               'training/compiled_rung_geometry.py', 'tools/plan_clearance_cycle.py', 'training/densify_reference.py')
    protocol = dict(continuation=args.continuous, maximum_knot_dt_s=.02 if args.continuous else None, parent=str(source), parent_sha256=sha(source), windows=windows, offset_m=offset,
        sources={p: sha(p) for p in sources}, position_tolerance_m=2e-5, orientation_tolerance_rad=1e-4,
        extra_joint_speed_limit_rad_s=.1, collision_sample_dt_s=.005,
        scene='Only rung_06 +2 mm X/+2 mm Z; other rods nominal. Not T12 foot-offset scene.',
        scope='Offline prototype only; full dynamics, independent prefix and policy acceptance absent; no training or baseline promotion.')
    write(args.output/'protocol.json', protocol)
    for relative in sources:
        dest=args.output/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(relative).read_bytes())
    base = CompiledPhysicalV2CycleEnv(plan=source, cases={'shift': {'rung_06': offset}})
    try:
        base.reset(options={'case':'shift'})
        e, parent_plan = base.env, base.plan
        original = densify_after(parent_plan,123.66) if args.continuous else parent_plan
        before = (e.data.qpos.copy(), e.data.qvel.copy(), float(e.data.time))
        result, details = refine_wholebody_hand_targets(e, original, offset_m=offset, windows=windows, continuation=args.continuous)
        np.savez_compressed(args.output/'candidate_knots.npz', **result)
        write(args.output/'node_diagnostics.json', details)
        print('Whole-body knots built; checking dense collision samples.', flush=True)
        collision = audit_plan(e, result, dt=.005)
        write(args.output/'candidate_geometry_audit.json', collision)
        d, times = mujoco.MjData(e.model), original['time']
        bodies=[e.model.body(n).id for n in ('hook','right_hook','left_ankle_roll_link','right_ankle_roll_link')]
        max_pos, max_rot, samples = 0., 0., 0
        for t in np.arange(123.66, times[-1]+1e-8, .005):
            j=min(max(int(np.searchsorted(times,t,side='right')-1),0),len(times)-2)
            u=float(np.clip((t-times[j])/(times[j+1]-times[j]),0,1))
            d.qpos[:]=(1-u)*original['qpos'][j]+u*original['qpos'][j+1]
            mujoco.mj_kinematics(e.model,d)
            target, rotations=d.xpos[bodies].copy(),d.xmat[bodies].copy()
            for b,key in enumerate(('lh','rh')):
                start,end=windows[key];g=float(np.clip((t-start)/(end-start),0,1))
                target[b]+=g**3*(10+g*(-15+6*g))*np.asarray(offset)
            d.qpos[:]=(1-u)*result['qpos'][j]+u*result['qpos'][j+1]
            mujoco.mj_kinematics(e.model,d)
            max_pos=max(max_pos,float(np.linalg.norm(d.xpos[bodies]-target,axis=1).max()))
            for before_rotation,after_rotation in zip(rotations,d.xmat[bodies]):
                angle=float(np.arccos(np.clip((np.trace(before_rotation.reshape(3,3).T@after_rotation.reshape(3,3))-1)/2,-1,1)))
                max_rot=max(max_rot,angle)
            samples+=1
        extra_speed=float(abs(np.diff(result['qpos'][:,e.qadr]-original['qpos'][:,e.qadr],axis=0)/np.diff(times)[:,None]).max())
        original_indices=np.searchsorted(original['time'],parent_plan['time'])
        densification_exact=all(np.array_equal(original[k][original_indices],parent_plan[k]) for k in ('qpos','time','active','phase','errors'))
        checks=dict(densification_preserves_original_knots=densification_exact,collision_and_native_limits=collision['passed'],dense_position=max_pos<2e-5,
            dense_orientation=max_rot<1e-4,extra_joint_speed=extra_speed<.1,
            prefix_exact=np.array_equal(result['qpos'][times<=123.66],original['qpos'][times<=123.66]),
            masks_and_timing_exact=all(np.array_equal(result[k],v) for k,v in original.items() if k not in ('qpos','errors')),
            base_orientation_exact=np.array_equal(result['qpos'][:,3:7],original['qpos'][:,3:7]),
            live_state_exact=np.array_equal(e.data.qpos,before[0]) and np.array_equal(e.data.qvel,before[1]) and e.data.time==before[2])
        summary=dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
            max_interpolated_position_error_m=max_pos,max_interpolated_rotation_error_rad=max_rot,
            maximum_extra_joint_speed_rad_s=extra_speed,interpolated_samples=samples,
            collision_failures=len(collision['failures']),candidate_sha256=sha(args.output/'candidate_knots.npz'),
            protocol_sha256=sha(args.output/'protocol.json'),scope=protocol['scope'])
        assert all(sha(p)==h for p,h in protocol['sources'].items()) and sha(source)==protocol['parent_sha256']
        write(args.output/'summary.json',summary)
        print(summary,flush=True)
    finally:
        base.close()


if __name__=='__main__':
    main()
