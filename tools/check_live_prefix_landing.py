"""Test live-prefix resets against a complete audited reference trajectory."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tools.compare_audited_closure_candidates import load_run,sha
from training.live_prefix_landing_env import LivePrefixLandingEnv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=['x+0.1_z15'])
    a=p.parse_args();torch.set_num_threads(1)
    verified=load_run(a.parent)
    assert set(a.cases)<=set(verified['cases'])
    a.output.mkdir(exist_ok=False)
    env=LivePrefixLandingEnv(a.parent,cases={k:verified['cases'][k] for k in a.cases},weight=0.,record=True)
    rows=[]
    try:
        for case in a.cases:
            obs,_=env.reset(options={'case':case})
            base=env.env.unwrapped
            arrays=env.prefix_trace
            while True:
                previous=obs.copy();action=env.teacher.predict(obs,deterministic=True)[0]
                obs,_,done,truncated,info=env.step(action)
                for key,value in dict(qpos=base.env.data.qpos.copy(),qvel=base.env.data.qvel.copy(),
                    time_s=float(base.env.data.time),observations=previous,actions=action.copy(),
                    torques=base.last_torque.copy(),base_rewards=info['base_reward']).items():arrays[key].append(value)
                if done or truncated:break
            arrays={k:np.asarray(v) for k,v in arrays.items()}
            arrays['waist_reference_trace']=np.asarray(env.waist_state['trace']).reshape(-1,10)
            arrays['adaptive_reference_trace']=np.asarray(env.reference_state['trace']).reshape(-1,21)
            with np.load(a.parent/f'after_{case}.npz') as old:
                exact={key:bool(np.array_equal(value,old['rewards' if key=='base_rewards' else key])) for key,value in arrays.items()}
                errors={key:float(np.max(np.abs(value-old['rewards' if key=='base_rewards' else key]),initial=0.)) for key,value in arrays.items()}
            target=a.output/f'{case}.npz';np.savez_compressed(target,**arrays)
            row=dict(case=case,passed=all(exact.values()) and info['success']==verified['rows'][case]['success'],
                     exact=exact,maximum_errors=errors,task_success=info['success'],trajectory_sha256=sha(target))
            rows.append(row);print(json.dumps(row),flush=True)
    finally:env.close()
    sources={key:sha(key) for key in ('training/live_prefix_landing_env.py','tools/check_live_prefix_landing.py')}
    result=dict(passed=all(row['passed'] for row in rows),rows=rows,parent_audit_sha256=verified['audit_sha256'],sources=sources,
        scope='Real physical prefix replay, no serialized state restore. Bitwise whole-record check on declared cases only. Not PPO training, reset throughput validation or all-case robustness.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    for name in sources:
        target=a.output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes())
    assert result['passed']


if __name__=='__main__':main()
