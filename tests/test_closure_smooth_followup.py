import copy
import unittest
from tools.compare_closure_smooth_followup import COMMON,validate_pair


class FollowupTests(unittest.TestCase):
    def pair(self):
        a={key:None for key in COMMON};a.update(training_cases={'a':{},'b':{}},parent_steps=98304,additional_steps_requested=98304,closure_cost_weight=0.,training_reward_contract='strict-v2-minus-terminal-closure-cost-v1')
        b=copy.deepcopy(a);b.update(closure_cost_weight=.02,training_reward_contract='strict-v2-minus-terminal-closure-smooth-cost-v1')
        s=dict(parent_steps=98304,additional_steps=98304,actual_steps=196608,cost_steps=98304,maximum_step_closure_cost=0.,total_training_closure_cost=0.,evaluation_pending=False)
        t=copy.deepcopy(s);t.update(maximum_step_closure_cost=.019,total_training_closure_cost=20.)
        return [a,b],[s,t]

    def test_complete_followup(self):validate_pair(*self.pair())

    def test_case_order_matters(self):
        m,s=self.pair();m[1]['training_cases']={'b':{},'a':{}}
        with self.assertRaises(ValueError):validate_pair(m,s)

    def test_changed_parent_or_reward_rejected(self):
        for key,value in [('parent_checkpoint_sha256','changed'),('training_reward_contract','unknown')]:
            m,s=self.pair();m[1][key]=value
            with self.assertRaises(ValueError):validate_pair(m,s)

    def test_partial_budget_rejected(self):
        m,s=self.pair();s[1]['additional_steps']-=1024
        with self.assertRaises(ValueError):validate_pair(m,s)


if __name__=='__main__':unittest.main()
