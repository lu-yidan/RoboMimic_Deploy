"""Compare online adapter to training observation and exact action/filter cadence."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.biased_load_transfer_residual_env import BiasedLoadTransferResidualEnv
from training.window_residual_policy import WindowResidualPolicy,load_observation

class Dummy:
    def __init__(self):self.calls=0
    def predict(self,obs,deterministic):
        self.calls+=1;return np.full(6,self.calls*.1),None

def main():
    manifest=json.loads((ROOT/'outputs/biased_load_transfer_ppo_32768_v1/manifest.json').read_text())
    env=BiasedLoadTransferResidualEnv(manifest['suite']);observations=[]
    for case in manifest['cases']:
        obs,_=env.reset(options=dict(case=case,normal_bias_y_deg=-3.))
        q,v,_=env.reference(env.env.data.time)
        observed=load_observation(env.env,q,env.env.data.time,np.zeros(6))
        assert np.array_equal(obs,observed);observations.append(case)
    model=Dummy();adapter=WindowResidualPolicy(env.env,model);original_q=q.copy();original_v=v.copy()
    for i in range(8):
        changed,_=adapter.command(q,v,49.+i*.005)
        assert model.calls==i//4+1
    assert np.array_equal(q,original_q) and np.array_equal(v,original_v)
    assert adapter.policy_records[0]['observation'][-6:]==[0.]*6
    assert np.allclose(adapter.policy_records[1]['observation'][-6:],[.1]*6)
    before=adapter.residual.copy();adapter.command(q,v,61.56)
    assert np.allclose(adapter.residual,.95*before) and model.calls==2
    assert np.array_equal(adapter.action,np.zeros(6))
    adapter.command(q,v,48.99);assert model.calls==2
    env.close();print(json.dumps(dict(passed=True,observation_exact_cases=observations,policy_50hz=True,filter_200hz=True,input_unmodified=True,smooth_decay=True)))

if __name__=='__main__':main()
