import copy
import unittest
from tools.compare_closure_margin_training import MATCHED,validate_pair


class ComparisonTests(unittest.TestCase):
    def data(self):
        m={key:None for key in MATCHED};m.update(parent_steps=98304,additional_steps_requested=98304,closure_cost_weight=0.)
        p=copy.deepcopy(m);p['closure_cost_weight']=.02
        s=dict(parent_steps=98304,additional_steps=98304,actual_steps=196608,cost_steps=98304,maximum_step_closure_cost=0.,total_training_closure_cost=0.,evaluation_pending=False)
        t=copy.deepcopy(s);t.update(maximum_step_closure_cost=.02,total_training_closure_cost=10.)
        return [m,p],[s,t]

    def test_complete_matched(self):
        validate_pair(*self.data())

    def test_partial_and_pending_rejected(self):
        for key,value in [('additional_steps',97280),('evaluation_pending',True),('cost_steps',1)]:
            m,s=self.data();s[1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_pair(m,s)

    def test_cost_in_control_rejected(self):
        m,s=self.data();s[0]['total_training_closure_cost']=.1
        with self.assertRaises(ValueError):validate_pair(m,s)

    def test_reference_mismatch_rejected(self):
        m,s=self.data();m[1]['reference_sha256']='different'
        with self.assertRaises(ValueError):validate_pair(m,s)


if __name__=='__main__':unittest.main()
