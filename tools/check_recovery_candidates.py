"""Verify candidate hashes, native PyTorch fixtures, and real deployment entry/run."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.validate_ft12k_sim import build_policy


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profiles',nargs='+',default=['ft_r0_9500','ft_r1_9999','ft_r2_9000','d4_9999']);parser.add_argument('--report',default='interface_validation.json');args=parser.parse_args()
    cfg=yaml.safe_load((ROOT/'policy/smp_recovery/config/smp_recovery.yaml').read_text())
    results={}
    for profile in args.profiles:
        entry=cfg['model_profiles'][profile]
        path=ROOT/'policy/smp_recovery/model'/entry['model_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['model_sha256']
        policy,state,output=build_policy(profile)
        fixture=np.load(ROOT/f'outputs/recovery_candidates/{profile}_parity.npz')
        actual=np.concatenate([policy._session.run(None,{'obs':x[None]})[0] for x in fixture['obs']])
        error=float(np.max(np.abs(actual-fixture['actions'])))
        np.testing.assert_allclose(actual,fixture['actions'],atol=5e-5,rtol=1e-5)
        state.q=policy.default_q.copy();state.dq=np.zeros(29,dtype=np.float32)
        state.root_ang_vel_b=np.zeros(3,dtype=np.float32);state.gravity_ori=np.array([0,0,-1],dtype=np.float32)
        policy.enter();policy.run()
        assert np.isfinite(output.actions).all()
        assert np.all(np.abs((output.actions-state.q)*policy.kps)<=policy.tau_limit+1e-4)
        result={'random_inputs':len(actual),'max_abs_error':error,'interface':[1,93,29], 'enter_run_passed':True}
        manifest_path=path.with_suffix('.manifest.json');manifest=json.loads(manifest_path.read_text())
        manifest['parity']=result;manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        results[profile]=result
    (ROOT/'outputs/recovery_candidates'/args.report).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
