"""Missing ramp commands must fail even when stored trace hashes are updated."""
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import unittest
from tools.audit_contact_trace_coverage import audit
from tools.validate_swing_reference import read,write,sha

ROOT=Path(__file__).resolve().parents[1]
SCRATCH=ROOT/'outputs/test_scratch'


class ContactCoverageTests(unittest.TestCase):
    def test_missing_ramp_command_rejected_with_updated_hashes(self):
        pilot=ROOT/'outputs/hook_admission_T12_two_20261029_v1'
        SCRATCH.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=SCRATCH) as temporary:
            out=Path(temporary).resolve();assert out.is_relative_to(SCRATCH.resolve())
            for name in ('manifest.json','hook_admission_protocol.json','hook_admission_record.json',
                         'hook_admission_audit.json','hook_admission_trace.json','reference.npz',
                         'physics_telemetry.npz','trajectory.npz'):
                shutil.copy2(pilot/name,out/name)
            rows=read(out/'hook_admission_trace.json')
            # Omit a ramp row after the confirmation window; the event itself
            # and its complete 50 ms feedback window remain in the record.
            i=next(i for i,r in enumerate(rows) if any(
                h['confirmed'] is not None and .15<r['time_s']-h['confirmed']<.3
                for h in r['hooks'].values()))
            rows.pop(i);write(out/'hook_admission_trace.json',rows)
            for name in ('hook_admission_record.json','hook_admission_audit.json'):
                value=read(out/name);value['trace_sha256']=sha(out/'hook_admission_trace.json')
                write(out/name,value)
            report=audit(out,out/'coverage.json')
            self.assertFalse(report['passed'])
            self.assertFalse(report['checks']['command_clock_complete'])
            self.assertEqual(report['expected_limited_commands'],235)
            self.assertEqual(report['recorded_limited_commands'],234)


if __name__=='__main__':unittest.main()
