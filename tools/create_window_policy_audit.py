"""Specialize independent reference audit for a frozen50Hz policy."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'tools/audit_window_constant.py').read_text(encoding='utf-8')
s=s.replace('import numpy as np','import numpy as np\nfrom stable_baselines3 import PPO\nimport torch')
s=s.replace('constant_residual_records.json','policy_control_records.json').replace('constant_online_reference.npz','policy_online_reference.npz').replace('constant_reference_audit.json','policy_reference_audit.json').replace("('constant__'+case)","('policy__'+case)")
needle="    errors={'action':0.,'offset':0.,'qref':0.,'vref':0.};residual=np.zeros(6)"
s=s.replace(needle,needle+'''
    policy_records=read(run/'policy_actions.json');torch.set_num_threads(1)
    model=PPO.load(run/'policy.zip',device='cpu')
    predicted=model.predict(np.asarray([r['observation'] for r in policy_records],dtype=np.float32),deterministic=True)[0]
    prediction_error=float(np.max(abs(predicted-np.asarray([r['action'] for r in policy_records]))))
    provenance=read(run/'policy_provenance.json')
    checkpoint_ok=hashlib.sha256((run/'policy.zip').read_bytes()).hexdigest()==provenance['sha256']==hashlib.sha256(Path(provenance['source']).read_bytes()).hexdigest()
    policy_i=0;window_steps=0;held=np.zeros(6);cadence_ok=True
    previous=np.zeros(6);history_ok=True
''')
needle="        t=row['time_s'];action=np.array([-.25,0.,0.,0.,0.,0.]) if 49.<=t<61.56 else np.zeros(6)"
assert needle in s
s=s.replace(needle,'''        t=row['time_s']
        if 49.<=t<61.56:
            if window_steps%4==0:
                recorded=policy_records[policy_i];policy_i+=1
                cadence_ok &= abs(recorded['time_s']-t)<1e-9
                history_ok &= np.allclose(recorded['observation'][-6:],previous,atol=1e-7,rtol=0.)
                held=np.asarray(recorded['action'])
            window_steps+=1
            if window_steps%4==0:previous=held.copy()
            action=held
        else:action=np.zeros(6)''')
s=s.replace('    result=dict(passed=all(checks.values())', '''    checks.update(checkpoint_unchanged=checkpoint_ok,policy_actions=prediction_error<2e-6,policy_cadence=bool(cadence_ok and policy_i==len(policy_records)),previous_action_observation=bool(history_ok))
    errors['batch_policy_prediction']=prediction_error
    result=dict(passed=all(checks.values())''')
s=s.replace('Fixed residual','Frozen policy residual').replace('fixed residual','frozen policy residual')
s=s.replace('Reference audit separate from task success.', 'Reference/action audit separate from task success. Ideal force observation channels are runtime logs, not independent force reconstruction.')
p=ROOT/'tools/audit_window_policy.py';assert not p.exists();p.write_text(s,encoding='utf-8')
