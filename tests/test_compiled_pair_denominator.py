"""Failed first cycles and baseline second cycles must not bias the paired set."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools import run_compiled_hook_pairs as suite


class PairDenominatorTests(unittest.TestCase):
    def test_failures_retained_and_candidate_not_gated_on_baseline_result(self):
        temporary_root=Path(__file__).resolve().parents[1]/'outputs'/'test_scratch'
        temporary_root.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=temporary_root) as temporary:
            root=Path(temporary).resolve()
            assert root.is_relative_to(temporary_root.resolve())
            single=root/'single';single.mkdir();out=root/'pair';out.mkdir()
            suite.write(single/'manifest.json',{})
            suite.write(single/'hook_uncertainty_protocol.json',{})
            suite.write(single/'compiled_reference_chain_audit.json',{})
            plan=root/'plan.npz';plan.write_bytes(b'test reference')
            cases={f'case_{i:02d}':{} for i in range(16)}
            suite.write(out/'protocol.json',dict(single_cycle_screen=str(single),
                single_manifest_sha256=suite.sha(single/'manifest.json'),
                hook_protocol_sha256=suite.sha(single/'hook_uncertainty_protocol.json'),
                plan=str(plan),plan_sha256=suite.sha(plan),sources={},cases=cases,scope='test'))
            attempted=[]
            def process(command,**kwargs):
                case=command[command.index('--case')+1];arm=command[command.index('--arm')+1]
                attempted.append((case,arm))
                return SimpleNamespace(returncode=1 if (case,arm)==('case_01','adopted_v2') else 0)
            def result(target,case,arm,code,log):
                return dict(case=case,arm=arm,passed=code==0,complete=code==0,status='synthetic_outcome')
            with patch.object(suite,'eligible',side_effect=lambda single,case:case!='case_00'), \
                 patch.object(suite.subprocess,'run',side_effect=process),patch.object(suite,'outcome',side_effect=result):
                suite.run(out,1)
            report=suite.read(out/'suite_summary.json')
            self.assertEqual(len(report['rows']),32)
            self.assertEqual(len(attempted),30)
            self.assertIn(('case_01','contact_confirmed'),attempted)
            self.assertEqual(report['passed_counts'],{'adopted_v2':14,'contact_confirmed':15})
            self.assertEqual(sum(r['status']=='first_cycle_gate_failed' for r in report['rows']),2)
            self.assertFalse(report['automatic_promotion'])
            self.assertFalse(report['all_records_successful'])


if __name__=='__main__':unittest.main()
