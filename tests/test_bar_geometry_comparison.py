import copy
import unittest
from tools.compare_bar_geometry_ablation import MATCHED,validate_pair


class ComparisonTest(unittest.TestCase):
    def pair(self):
        a={k:'same' for k in MATCHED};a.update(geometry_mode='blind',requested_steps=98304)
        b=copy.deepcopy(a);b['geometry_mode']='oracle'
        s=dict(actual_steps=98304,additional_steps=98304,parent_steps=0,evaluation_pending=False)
        return [a,b],[s,copy.deepcopy(s)]

    def test_valid_pair(self):validate_pair(*self.pair())

    def test_rejects_same_short_budget(self):
        m,s=self.pair()
        for r in s:r.update(actual_steps=1024,additional_steps=1024)
        with self.assertRaises(ValueError):validate_pair(m,s)

    def test_rejects_different_cases_and_pending(self):
        m,s=self.pair();m[1]['training_cases']='different'
        with self.assertRaises(ValueError):validate_pair(m,s)
        m,s=self.pair();s[1]['evaluation_pending']=True
        with self.assertRaises(ValueError):validate_pair(m,s)


if __name__=='__main__':unittest.main()
