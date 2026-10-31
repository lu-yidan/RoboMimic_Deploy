"""Check the zero-weight factory against its frozen pre-handoff implementation.

This is a wrapper/factory regression check with a deterministic test environment,
not PPO training and not a substitute for the recorded physical replay audits.
"""
import ast
import copy
import json
from pathlib import Path
from unittest.mock import patch
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from training.balanced_case_reset import BalancedCaseReset
from tools import train_calibrated_landing as current
from tools.prepare_calibrated_training import check_terminal_margin
from tools.validate_swing_reference import ROOT,write,sha


class DeterministicProbe(gym.Env):
    def __init__(self,parent,*,cases,weight,mode):
        self.cases=cases;self.weight=weight;self.mode=mode
        self.observation_space=gym.spaces.Box(-10,10,(2,),dtype=np.float32)
        self.action_space=gym.spaces.Box(-1,1,(1,),dtype=np.float32)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed);self.case=options['case'];self.t=0
        self.value=float(list(self.cases).index(self.case))
        return np.array([self.value,0],np.float32),dict(case=self.case)
    def step(self,action):
        self.t+=1;self.value+=float(action[0])
        return np.array([self.value,self.t],np.float32),float(self.value*self.weight),self.t==3,False,dict(case=self.case)


def trace(env):
    result=[]
    for episode in range(12):
        obs,info=env.reset(seed=123 if episode==0 else None)
        result.append([obs.tolist(),info['case']])
        for action in (.1,-.2,.3):
            obs,reward,done,truncated,info=env.step(np.array([action],np.float32))
            result.append([obs.tolist(),reward,done,truncated,info['case']])
    chain=[];node=env
    while node is not None:chain.append(type(node).__name__);node=getattr(node,'env',None)
    env.close();return dict(trace=result,wrapper_chain=chain)


def main():
    out=ROOT/'outputs/terminal_margin_handoff_review_20261029_v1';out.mkdir(exist_ok=False)
    source=ROOT/'outputs/calibrated_geometry_training_baseline_20261027_v1/source_snapshot/tools/train_calibrated_landing.py'
    tree=ast.parse(source.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_env')
    namespace=dict(torch=torch,BoundaryLandingEnv=DeterministicProbe,BalancedCaseReset=BalancedCaseReset,Monitor=Monitor,Path=Path)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    protocol=dict(parent='unused-test-parent',training_cases={'a':{},'b':{}},partitions=[['a','b']],quality_bonus_weight=.1)
    old=trace(namespace['make_env'](protocol,'average',out/'old',0));rows=[]
    for label,margin in [('absent','absent'),('null',None),('empty',{}),('zero',dict(weight=0.,scale_mrad=1.5))]:
        candidate=copy.deepcopy(protocol)
        if label!='absent':candidate['terminal_margin']=margin
        with patch.object(current,'BoundaryLandingEnv',DeterministicProbe):
            actual=trace(current.make_env(candidate,'average',out/label,0))
        rows.append(dict(configuration=label,passed=actual==old,wrapper_chain=actual['wrapper_chain']))
    check_terminal_margin(0.,1.5);check_terminal_margin(6.,1.5)
    low_weight_issue=False
    try:check_terminal_margin(.1,1.5)
    except AssertionError:low_weight_issue=True
    report=dict(passed=all(r['passed'] for r in rows),rows=rows,zero_and_six_weight_scalar_checks_passed=True,
        low_positive_weight_gate_rejected=low_weight_issue,
        old_factory_source=str(source),old_factory_sha256=sha(source),current_factory_sha256=sha(ROOT/'tools/train_calibrated_landing.py'),
        module_sha256=sha(ROOT/'training/terminal_margin_reward.py'),review_source_sha256=sha(__file__),
        findings=['Zero/default weight does not insert TerminalMarginReward; seeded case order, observations, actions and rewards match the frozen factory in this regression check.',
            'prepare_calibrated_training.py has no pre-change snapshot in the handoff or baseline; a complete five-file byte-level historical equivalence claim is unsupported.',
            'check_terminal_margin hard-codes gain > 0.5 even for otherwise valid small positive weights; weight 0.1 is rejected by that diagnostic gate.',
            'The optional reward still encodes the old 40 mrad/10 mm margin and uses the final sample, not worst-case last-two-second metrics. Keep it disabled under adopted v2; no evidence of useful retraining gain.'],
        scope='Static five-file review plus deterministic factory regression and scalar reward checks. No training, no new reward adoption, and no new physical equivalence rollout.')
    write(out/'summary.json',report);(out/'review_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report),flush=True)
    assert report['passed']


if __name__=='__main__':main()
