"""Check actual PPO parameter updates separately from task performance."""
import argparse,json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from tools.compare_audited_closure_candidates import sha
from tools.continue_vector_full_cycle_training import state_difference


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();torch.set_num_threads(1)
    m=json.loads((a.run/'manifest.json').read_text());s=json.loads((a.run/'summary.json').read_text());parent=Path(m['parent_experiment'])
    pm=json.loads((parent/'manifest.json').read_text());protocol=json.loads((a.run/'matched_protocol.json').read_text())
    before=PPO.load(a.run/'policy_before.zip',device='cpu');after=PPO.load(a.run/'policy.zip',device='cpu');original=PPO.load(parent/'policy.zip',device='cpu')
    initial=before.policy.state_dict();final=after.policy.state_dict()
    groups={}
    for name in ('actor_and_shared','deterministic_actor_and_shared','critic'):
        keys=[k for k in initial if k.startswith(('mlp_extractor.value_net.','value_net.'))==(name=='critic') and (name!='deterministic_actor_and_shared' or k!='log_std')]
        groups[name]=state_difference({k:initial[k] for k in keys},{k:final[k] for k in keys})
    checks=dict(initial_parameters_exact=state_difference(initial,original.policy.state_dict())==0,
        initial_optimizer_exact=state_difference(before.policy.optimizer.state_dict(),original.policy.optimizer.state_dict())==0,
        finite_budget_completed=s['additional_steps']==protocol['additional_steps_per_arm']==32768,
        timestep_accounting=after.num_timesteps-before.num_timesteps==s['additional_steps'],
        calibration_unchanged=all(m[k]==pm[k] for k in ('policy_adapter','closure_action_bias','controller_modifier','adaptive_reference_command','prefix_checkpoint_sha256')),
        parent_runtime_sources_unchanged=all(m['sources'].get(f)==h for f,h in pm['sources'].items() if f.replace('\\','/').startswith('training/')),
        parameters_updated=all(v>0 for v in groups.values()),checkpoint_matches_summary=sha(a.run/'policy.zip')==s['checkpoint_sha256'])
    result=dict(passed=all(checks.values()),checks=checks,maximum_parameter_changes=groups,
        before_steps=before.num_timesteps,after_steps=after.num_timesteps,additional_steps=s['additional_steps'],
        before_sha256=sha(a.run/'policy_before.zip'),after_sha256=sha(a.run/'policy.zip'),parent_sha256=sha(parent/'policy.zip'),
        manifest_sha256=sha(a.run/'manifest.json'),source_sha256=sha(__file__),scope='Optimizer/parameter and fixed-calibration checks only. Task success and convergence must be assessed separately.')
    target=a.run/'parameter_update_audit.json';assert not target.exists()
    target.write_text(json.dumps(result,indent=2));(a.run/'parameter_update_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result),flush=True);assert result['passed']


if __name__=='__main__':main()
