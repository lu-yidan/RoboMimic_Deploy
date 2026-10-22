"""Build isolated frozen-policy full-cycle runner from constant experiment."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'deploy_mujoco/run_window_constant_cycle.py').read_text(encoding='utf-8')
s=s.replace('deploy_mujoco/run_window_constant_cycle.py','deploy_mujoco/run_window_policy_cycle.py')
s=s.replace('import csv','from stable_baselines3 import PPO\nimport torch\nfrom training.window_residual_policy import WindowResidualPolicy\nimport csv',1)
s=s.replace("extra_sources=[", "extra_sources=['training/window_residual_policy.py',",1)
s=s.replace('    residual=np.zeros(6)\n    residual_records=[]', '''    policy_path=ROOT/'outputs/biased_load_transfer_ppo_32768_v1/policy.zip'
    policy_audit=json.loads((policy_path.parent/'continuation_audit.json').read_text())
    if not policy_audit['passed']:raise ValueError('Policy continuation audit failed')
    torch.set_num_threads(1)
    policy=WindowResidualPolicy(env,PPO.load(policy_path,device='cpu'))
    shutil.copy2(policy_path,out/'policy.zip')
    (out/'policy_provenance.json').write_text(json.dumps(dict(source=str(policy_path),sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),scope='Frozen local residual in49..61.56s, smooth decay afterwards. Rest of cycle remains scripted, not full-cycle learned locomotion.')))''')
start=s.index('        # Same action/filter')
end=s.index('\n        if guard:',start)
s=s[:start]+"        qref,vref=policy.command(qref,vref,float(d.time))"+s[end:]
s=s.replace("(out/'constant_residual_records.json').write_text(json.dumps(residual_records))", "(out/'policy_control_records.json').write_text(json.dumps(policy.control_records))\n    (out/'policy_actions.json').write_text(json.dumps(policy.policy_records))")
s=s.replace('constant_online_reference.npz','policy_online_reference.npz')
s=s.replace('trained_policy=False,','trained_policy=True,learned_full_cycle=False,')
p=ROOT/'deploy_mujoco/run_window_policy_cycle.py';assert not p.exists();p.write_text(s,encoding='utf-8')
s=(ROOT/'tools/run_window_constant_experiment.py').read_text(encoding='utf-8')
s=s.replace('run_window_constant_cycle','run_window_policy_cycle').replace("name='constant__'+case","name='policy__'+case")
s=s.replace('constant_action=[-.25,0,0,0,0,0]',"checkpoint='outputs/biased_load_transfer_ppo_32768_v1/policy.zip'")
s=s.replace('Constant action selected on one known hardcase, ramps to zero after61.56s via same filter; not learned.', 'Frozen32768-step local policy, ramps residual to zero after61.56s; outside window scripted. No full-cycle learned-policy claim.')
p=ROOT/'tools/run_window_policy_experiment.py';assert not p.exists();p.write_text(s,encoding='utf-8')
