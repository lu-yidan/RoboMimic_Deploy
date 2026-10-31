"""Zero-step preflight for the separately queued new-reference physics pilot."""
import argparse
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.compiled_rung_geometry import CompiledPhysicalV2CycleEnv
from training.cycle_relative_bar_observation import CycleRelativeBarObservation
from training.closure_action_bias_policy_v2 import ClosureActionBiasV2Policy
from tools.validate_swing_reference import read, write, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=Path('outputs/wholebody_hand_dynamic_protocol_20261029_v1/protocol.json')
    declared=read(protocol);parent=Path(declared['candidate']);m=read(parent/'manifest.json')
    plans=[Path('outputs/v2_swing_two_reference_20261029_v1/plan.npz'),Path(declared['plan'])]
    assert sha(plans[1])==declared['plan_sha256']
    assert read(parent/'compiled_reference_chain_audit.json')['passed']
    snapshots=[]
    for plan in plans:
        base=CompiledPhysicalV2CycleEnv(plan=plan,cases={declared['case']:m['evaluation_cases'][declared['case']]},
            normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['force_history_memory'])
        observer=CycleRelativeBarObservation(base,mode=m['geometry_mode'])
        try:
            observation,_=observer.reset(seed=20261022,options={'case':declared['case']})
            samples=[base.reference(t) for t in (0.,70.2,113.66,119.66,123.66)]
            snapshots.append(dict(qpos=base.env.data.qpos.copy(),qvel=base.env.data.qvel.copy(),
                observation=observation.copy(),samples=samples,geometry=base.variant['geometry_sha256'],time=float(base.env.data.time)))
        finally:base.close()
    old,new=snapshots
    checks={name:np.array_equal(old[name],new[name]) for name in ('qpos','qvel','observation')}
    checks.update(geometry_exact=old['geometry']==new['geometry'],zero_physics_time=old['time']==new['time']==0.,
        first_cycle_and_before_change_reference_exact=all(np.array_equal(x,y) for before,after in zip(old['samples'][:-1],new['samples'][:-1]) for x,y in zip(before,after)),
        change_onset_pose_exact=np.array_equal(old['samples'][-1][0],new['samples'][-1][0]),
        change_onset_contact_schedule_exact=all(np.array_equal(x,y) for x,y in zip(old['samples'][-1][2:],new['samples'][-1][2:])))
    # At the first changed interval, the piecewise-linear reference's forward
    # velocity already changes even though its starting pose is unchanged.
    onset_velocity_difference=float(abs(old['samples'][-1][1]-new['samples'][-1][1]).max())
    torch.set_num_threads(1)
    model=ClosureActionBiasV2Policy(PPO.load(parent/'policy.zip',device='cpu'),m)
    action_old=model.predict(old['observation'],deterministic=True)[0]
    action_new=model.predict(new['observation'],deterministic=True)[0]
    checks['initial_policy_action_exact']=np.array_equal(action_old,action_new)
    report=dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
        protocol_sha256=sha(protocol),parent_manifest_sha256=sha(parent/'manifest.json'),
        checkpoint_sha256=sha(parent/'policy.zip'),source_sha256=sha(__file__),physics_steps=0,
        change_onset_velocity_difference=onset_velocity_difference,
        scope='Freshly compiled initial scene, observations, initial actor action and sampled unchanged reference prefix only. No dynamics, full prefix replay or task-success claim.')
    write(a.output/'summary.json',report);(a.output/'preflight_source.py').write_bytes(Path(__file__).read_bytes())
    print(report,flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
