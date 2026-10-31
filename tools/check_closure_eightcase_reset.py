"""Verify pooled native-state restoration across eight geometry cases."""
import json
from pathlib import Path
import numpy as np
from training.closure_curriculum_env_v3 import ClosureCurriculumEnvV3
from tools.continue_vector_full_cycle_training import sha


def main():
    cache=Path('outputs/closure_eightcase_prefix_cache_20261024_v1');out=Path('outputs/closure_eightcase_reset_gate_20261024_v1');out.mkdir(exist_ok=False)
    summary=json.loads((cache/'summary.json').read_text());assert summary['passed']
    env=ClosureCurriculumEnvV3(cache);rows=[]
    try:
        for row in summary['rows']:
            case=row['case'];source_cache=Path(row['source_cache_directory']);protocol=json.loads((source_cache/'protocol.json').read_text());parent=Path(protocol['parent'])
            path=parent/f'after_{case}.npz';assert sha(path)==row['source_trajectory_sha256'];index=4983
            with np.load(path) as z:
                obs,_=env.reset(options={'case':case});base=env.env.unwrapped
                assert np.array_equal(obs,z['observations'][index]) and np.array_equal(base.env.data.qpos,z['qpos'][index]) and np.array_equal(base.env.data.qvel,z['qvel'][index])
                steps=700 if case in ('z15.1','x-0.1_z15','x+0.1_z15') else 1
                difference=dict(qpos=0.,qvel=0.,torques=0.,base_reward=0.)
                for i in range(index,index+steps):
                    obs,reward,done,truncated,info=env.step(z['actions'][i])
                    for key,actual,expected in [('qpos',base.env.data.qpos,z['qpos'][i+1]),('qvel',base.env.data.qvel,z['qvel'][i+1]),('torques',base.last_torque,z['torques'][i]),('base_reward',info['base_reward'],z['rewards'][i])]:
                        difference[key]=max(difference[key],float(np.max(abs(actual-expected))))
                for key,tolerance in dict(qpos=1e-7,qvel=1e-5,torques=.001,base_reward=1e-6).items():assert difference[key]<=tolerance
                if steps==700:assert done and not info['success']
                else:assert not done
                result=dict(case=case,checked_steps=steps,differences=difference,initial_state_exact=True,passed=True)
                rows.append(result);print(json.dumps(result),flush=True)
        (out/'summary.json').write_text(json.dumps(dict(passed=True,rows=rows,cache_summary_sha256=sha(cache/'summary.json'),source_sha256=sha(__file__),environment_sha256=sha('training/closure_curriculum_env_v3.py'),
            scope='All eight reset states and first transitions checked against original continuous runs; three added geometries additionally replay full suffix. Original actor still fails added cases, as expected. This is reset integrity, not a successful policy.'),indent=2))
    finally:env.close()

if __name__=='__main__':main()
