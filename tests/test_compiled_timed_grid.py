"""Third-arm denominator and independent timing-audit regressions."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import shutil
import unittest
from unittest.mock import patch
from tools import run_compiled_timed_grid as suite
from tools.audit_compiled_timed_hook import audit_timing
from tools.audit_timed_trace_coverage import audit as audit_coverage

ROOT=Path(__file__).resolve().parents[1]
SCRATCH=ROOT/'outputs/test_scratch'


class TimedGridTests(unittest.TestCase):
    def test_all_sixteen_retained_without_reading_second_cycle_results(self):
        SCRATCH.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=SCRATCH) as temporary:
            root=Path(temporary).resolve();assert root.is_relative_to(SCRATCH.resolve())
            single=root/'single';single.mkdir();out=root/'timed';out.mkdir()
            pilot=root/'pilot';pilot.mkdir()
            for target in (single/'manifest.json',single/'compiled_reference_chain_audit.json',
                           pilot/'timed_loading_audit.json',root/'declaration.json'):
                suite.write(target,{})
            plan=root/'plan.npz';plan.write_bytes(b'synthetic reference')
            suite.write(out/'protocol.json',dict(single_cycle_screen=str(single),
                single_manifest_sha256=suite.sha(single/'manifest.json'),
                declaration=str(root/'declaration.json'),declaration_sha256=suite.sha(root/'declaration.json'),
                pilot=str(pilot),pilot_audit_sha256=suite.sha(pilot/'timed_loading_audit.json'),
                plan=str(plan),plan_sha256=suite.sha(plan),sources={},scope='synthetic test',
                cases={f'case_{i:02d}':{} for i in range(16)}))
            attempted=[]
            def process(command,**kwargs):
                case=command[command.index('--case')+1];attempted.append(case)
                self.assertEqual(command[command.index('--arm')+1],'timed_open_loop')
                return SimpleNamespace(returncode=1 if case=='case_01' else 0)
            def result(target,case,arm,code,log):
                return dict(case=case,arm=arm,passed=code==0,complete=code==0,status='synthetic')
            with patch.object(suite,'eligible',side_effect=lambda single,case:case!='case_00'), \
                 patch.object(suite.subprocess,'run',side_effect=process),patch.object(suite,'outcome',side_effect=result):
                suite.run(out,1)
            report=suite.read(out/'suite_summary.json')
            self.assertEqual(len(report['rows']),16);self.assertEqual(len(attempted),15)
            self.assertEqual(report['passed_count'],14)
            self.assertEqual(sum(r['status']=='first_cycle_gate_failed' for r in report['rows']),1)
            self.assertFalse(report['automatic_promotion'])

    def check_saved_pilot(self,tamper):
        pilot=ROOT/'outputs/timed_hook_T12_two_20261029_v1'
        SCRATCH.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=SCRATCH) as temporary:
            out=Path(temporary).resolve();assert out.is_relative_to(SCRATCH.resolve())
            for name in ('manifest.json','timed_loading_protocol.json','timed_loading_record.json',
                'v2_multicycle_audit.json','physics_telemetry.npz','trajectory.npz','reference.npz','timed_loading_trace.json'):
                shutil.copy2(pilot/name,out/name)
            if tamper:
                trace=suite.read(out/'timed_loading_trace.json')
                changed=False
                for row in trace:
                    for hook in row['hooks'].values():
                        if hook['released'] is not None:
                            hook['released']+=.005;changed=True;break
                    if changed:break
                self.assertTrue(changed);suite.write(out/'timed_loading_trace.json',trace)
                record=suite.read(out/'timed_loading_record.json')
                record['trace_sha256']=suite.sha(out/'timed_loading_trace.json')
                suite.write(out/'timed_loading_record.json',record)
            result=audit_timing(out)
            self.assertEqual(result['passed'],not tamper)
            self.assertEqual(result['checks']['release_on_declared_clock'],not tamper)

    def test_saved_nominal_timing_accepted(self):self.check_saved_pilot(False)
    def test_wrong_release_clock_rejected_even_when_record_hash_updated(self):self.check_saved_pilot(True)

    def test_missing_middle_command_is_rejected(self):
        pilot=ROOT/'outputs/timed_hook_T12_two_20261029_v1'
        SCRATCH.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=SCRATCH) as temporary:
            out=Path(temporary).resolve();assert out.is_relative_to(SCRATCH.resolve())
            for name in ('manifest.json','timed_loading_protocol.json','timed_loading_record.json',
                'v2_multicycle_audit.json','physics_telemetry.npz','trajectory.npz','reference.npz','timed_loading_trace.json'):
                shutil.copy2(pilot/name,out/name)
            rows=suite.read(out/'timed_loading_trace.json');rows.pop(20)
            suite.write(out/'timed_loading_trace.json',rows)
            record=suite.read(out/'timed_loading_record.json')
            record['trace_sha256']=suite.sha(out/'timed_loading_trace.json')
            suite.write(out/'timed_loading_record.json',record)
            report=audit_coverage(out,out/'coverage.json')
            self.assertFalse(report['passed'])
            self.assertFalse(report['checks']['command_clock_complete'])
            self.assertEqual(report['expected_limited_commands'],235)
            self.assertEqual(report['recorded_limited_commands'],234)


if __name__=='__main__':unittest.main()
