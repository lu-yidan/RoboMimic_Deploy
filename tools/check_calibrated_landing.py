"""Test live-prefix resets against a complete audited reference trajectory."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tools.compare_audited_closure_candidates import load_run,sha
from training.calibrated_landing_env import CalibratedLandingEnv as BoundaryLandingEnv
from training.terminal_margin_reward import TerminalMarginReward, terminal_bonus, _strict_env


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['average','bottleneck'],required=True)
    p.add_argument('--cases',nargs='+',default=['x+0.1_z15'])
    p.add_argument('--terminal-margin-weight',type=float,default=0.)
    p.add_argument('--terminal-margin-scale-mrad',type=float,default=1.5)
    a=p.parse_args();torch.set_num_threads(1)
    verified=load_run(a.parent);assert set(a.cases)<=set(verified['cases'])
    a.output.mkdir(exist_ok=False)
    inner=BoundaryLandingEnv(a.parent,cases={k:verified['cases'][k] for k in a.cases},weight=.1,record=True,mode=a.mode)
    env=TerminalMarginReward(inner,weight=a.terminal_margin_weight,
                             scale_mrad=a.terminal_margin_scale_mrad) if a.terminal_margin_weight>0 else inner
    rows=[]
    try:
        for case in a.cases:
            obs,_=env.reset(options={'case':case})
            base=inner.env.unwrapped
            arrays=inner.prefix_trace
            while True:
                previous=obs.copy();action=inner.teacher.predict(obs,deterministic=True)[0]
                obs,reward,done,truncated,info=env.step(action)
                components=np.array([max(0.,info['current_closure_rad']-.032)/.008,
                                     max(0.,abs(info['current_rise_m']-.25)-.006)/.004])
                values=1./(1.+components**2)
                expected=.1*info['closure_cost_ramp']*(values.mean() if a.mode=='average' else values.min())
                terminal_expected=0.
                if a.terminal_margin_weight>0 and (done or truncated):
                    strict=_strict_env(inner)
                    complete=len(strict.cycle_results)==len(strict.ends)
                    terminal_expected=terminal_bonus(info['current_closure_rad'],info['current_rise_m'],
                        weight=a.terminal_margin_weight,scale_mrad=a.terminal_margin_scale_mrad,completed=complete)
                    assert abs(info.get('terminal_margin_bonus',terminal_expected)-terminal_expected)<1e-12
                assert abs(info['closure_quality_bonus']-expected)<1e-14
                assert abs(reward-(info['base_reward']+expected+terminal_expected))<1e-12
                for key,value in dict(qpos=base.env.data.qpos.copy(),qvel=base.env.data.qvel.copy(),
                    time_s=float(base.env.data.time),observations=previous,actions=inner.last_executed_action.copy(),
                    torques=base.last_torque.copy(),base_rewards=info['base_reward']).items():arrays[key].append(value)
                if done or truncated:break
            arrays={k:np.asarray(v) for k,v in arrays.items()}
            arrays['waist_reference_trace']=np.asarray(inner.waist_state['trace']).reshape(-1,10)
            arrays['adaptive_reference_trace']=np.asarray(inner.reference_state['trace']).reshape(-1,21)
            with np.load(a.parent/f'after_{case}.npz') as old:
                exact={key:bool(np.array_equal(value,old['rewards' if key=='base_rewards' else key])) for key,value in arrays.items()}
                errors={key:float(np.max(np.abs(value-old['rewards' if key=='base_rewards' else key]),initial=0.)) for key,value in arrays.items()}
            target=a.output/f'{case}.npz';np.savez_compressed(target,**arrays)
            row=dict(case=case,passed=all(exact.values()) and info['success']==verified['rows'][case]['success'],
                     exact=exact,maximum_errors=errors,task_success=info['success'],trajectory_sha256=sha(target))
            rows.append(row);print(json.dumps(row),flush=True)
    finally:env.close()
    sources={key:sha(key) for key in ('training/live_prefix_landing_env.py','training/boundary_landing_env.py','training/closure_quality_reward.py','training/calibrated_landing_env.py','tools/check_calibrated_landing.py')}
    result=dict(reward_mode=a.mode, reward_reconstruction_passed=True, passed=all(row['passed'] for row in rows),rows=rows,parent_audit_sha256=verified['audit_sha256'],sources=sources,
        scope='Raw PPO actions pass through the declared fixed v2 calibration. Real physical prefix replay, no serialized state restore. Bitwise whole-record check on declared cases only. Not PPO training, reset throughput validation or all-case robustness.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    for name in sources:
        target=a.output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes())
    assert result['passed']


if __name__=='__main__':main()
