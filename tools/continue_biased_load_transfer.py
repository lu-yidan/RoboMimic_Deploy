"""Continue a fully audited PPO checkpoint with explicitly reused evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from training.biased_load_transfer_residual_env import BiasedLoadTransferResidualEnv
from tools.train_biased_load_transfer import evaluate


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--additional-steps',type=int,default=24576);a=p.parse_args()
    parent=a.parent.resolve();a.out=a.out.resolve()
    old=json.loads((parent/'manifest.json').read_text());prior=json.loads((parent/'summary.json').read_text())
    audit=json.loads((parent/'independent_audit.json').read_text())
    if not audit['integrity_passed']:raise ValueError('Parent must pass independent audit first')
    if a.additional_steps<256 or a.additional_steps%256:raise ValueError('Additional steps must be a positive multiple of256')
    if sha(parent/'policy.zip')!=prior['checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
    for rel,digest in old['sources'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Source changed since parent: '+rel)
    suite=Path(old['suite'])
    for name,digest in old['prefixes'].items():
        if sha(suite/name/'trajectory.npz')!=digest:raise ValueError('Prefix changed')
    base=BiasedLoadTransferResidualEnv(suite);torch.set_num_threads(1)
    a.out.mkdir(parents=True,exist_ok=False)
    sources=dict(old['sources']);sources['tools/continue_biased_load_transfer.py']=sha(Path(__file__))
    for rel in sources:
        dest=a.out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    reused={}
    for destination,label in [('zero','zero'),('before','after')]:
        for row in prior[label]:
            src=parent/f"{label}_{row['scenario']}.npz";dest=a.out/f"{destination}_{row['scenario']}.npz"
            shutil.copy2(src,dest);reused[dest.name]=dict(source=str(src),sha256=sha(src))
        (a.out/(destination+'.json')).write_text(json.dumps(prior[label],indent=2))
    shutil.copy2(parent/'policy.zip',a.out/'policy_before.zip')
    manifest=dict(old,sources=sources,steps=prior['actual_steps']+a.additional_steps,parent_run=str(parent),parent_checkpoint_sha256=prior['checkpoint_sha256'],additional_steps=a.additional_steps,reused_evaluations=reused,continuation='Policy/optimizer resume; environment reset and seeded RNG restart. Not bitwise continuation of an interrupted simulator. Zero reused from audited parent, before is audited parent after; hashes recorded.')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    env=Monitor(base,str(a.out/'episodes'),info_keywords=('scenario','normal_bias_y_deg','minimum_native_joint_margin_rad'))
    model=PPO.load(parent/'policy.zip',env=env,device='cpu');model.set_random_seed(old['seed'])
    model.set_logger(configure(str(a.out),['csv']))
    initial_steps=model.num_timesteps;start=time.perf_counter()
    model.learn(total_timesteps=a.additional_steps,reset_num_timesteps=False)
    elapsed=time.perf_counter()-start;model.save(a.out/'policy')
    after=evaluate(model,suite,a.out,'after',old['seed'])
    loaded=PPO.load(a.out/'policy.zip',device='cpu');difference=0.
    for row in after:
        z=dict(np.load(a.out/f"after_{row['scenario']}.npz"))
        difference=max(difference,float(np.max(abs(model.predict(z['observations'],deterministic=True)[0]-loaded.predict(z['observations'],deterministic=True)[0]))))
    if difference:raise AssertionError('Checkpoint reload changed actions')
    result=dict(training_completed=True,actual_steps=model.num_timesteps,parent_steps=initial_steps,additional_steps=model.num_timesteps-initial_steps,training_wall_s=elapsed,zero=prior['zero'],before=prior['after'],after=after,checkpoint_reload_max_difference=difference,checkpoint_sha256=sha(a.out/'policy.zip'),learned_full_cycle=False,multi_seed_validated=False)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2));env.close()
    print(json.dumps(dict(training_completed=True,actual_steps=model.num_timesteps,additional_steps=model.num_timesteps-initial_steps)),flush=True)


if __name__=='__main__':main()
