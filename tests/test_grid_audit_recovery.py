"""Audit recovery must use copies and never rerun failed/partial physics."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools import recover_completed_grid_audits as recovery

ROOT=Path(__file__).resolve().parents[1]


class GridAuditRecoveryTest(unittest.TestCase):
    def setUp(self):
        scratch=ROOT/'outputs/test_scratch';scratch.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='audit recovery ',dir=scratch)
        self.root=Path(self.temp.name).resolve()
        assert self.root.is_relative_to(scratch.resolve())
        self.pair,self.timed=self.root/'paired',self.root/'timed'
        for folder in (self.pair,self.timed):
            folder.mkdir();recovery.write(folder/'protocol.json',dict(sources={}))
        self.run=self.pair/'run';self.run.mkdir()
        (self.run/'trajectory.npz').write_bytes(b'fixture trajectory')
        (self.run/'physics_telemetry.npz').write_bytes(b'fixture telemetry')
        recovery.write(self.run/'summary.json',dict(passed=True,success=True,time_s=227.32))
        self.log=self.pair/'original.log'
        self.log.write_text("tools.audit_compiled_hook_pair returned non-zero exit status 3221225480")
        self.row=dict(case='test',arm='adopted_v2',passed=False,complete=True,run=str(self.run),log=str(self.log),
            trajectory_sha256=recovery.sha(self.run/'trajectory.npz'),telemetry_sha256=recovery.sha(self.run/'physics_telemetry.npz'))

    def tearDown(self):
        assert self.root.is_relative_to((ROOT/'outputs/test_scratch').resolve())
        self.temp.cleanup()

    def summaries(self):
        for folder,rows in ((self.pair,[self.row]),(self.timed,[])):
            recovery.write(folder/'suite_summary.json',dict(finished=True,sources_unchanged=True,
                protocol_sha256=recovery.sha(folder/'protocol.json'),rows=rows))

    def invoke(self,output):
        with patch('sys.argv',['recover','--paired',str(self.pair),'--timed',str(self.timed),'--output',str(output)]):
            recovery.main()

    def test_partial_physical_failure_and_other_error_not_retried(self):
        for reason in ('partial','physical_failure','other_error'):
            self.row['complete']=reason!='partial'
            recovery.write(self.run/'summary.json',dict(passed=reason!='physical_failure',success=reason!='physical_failure',time_s=227.32))
            self.log.write_text('unrelated failure' if reason=='other_error' else 'tools.audit_compiled_hook_pair non-zero exit status 3221225480')
            self.summaries()
            out=self.root/reason
            with self.subTest(reason=reason), patch.object(recovery.subprocess,'run',side_effect=AssertionError('Must not launch any process')):
                self.invoke(out)
            self.assertEqual(recovery.read(out/'summary.json')['attempted'],0)

    def test_one_audit_on_copy_original_files_unchanged(self):
        self.summaries()
        original={p.name:recovery.sha(p) for p in self.run.iterdir()}
        def fake_audit(command,**kwargs):
            self.assertEqual(command[1:3],['-m','tools.audit_compiled_hook_pair'])
            copied=Path(command[-1]);self.assertNotEqual(copied,self.run)
            recovery.write(copied/'compiled_pair_audit.json',dict(passed=True,
                trajectory_sha256=original['trajectory.npz'],telemetry_sha256=original['physics_telemetry.npz']))
            return SimpleNamespace(returncode=0)
        out=self.root/'recovered'
        with patch.object(recovery.subprocess,'run',side_effect=fake_audit) as subprocess:
            self.invoke(out)
        self.assertEqual(subprocess.call_count,1)
        report=recovery.read(out/'summary.json')
        self.assertEqual(report['passed'],1)
        self.assertFalse((self.run/'compiled_pair_audit.json').exists())
        self.assertEqual({p.name:recovery.sha(p) for p in self.run.iterdir()},original)
        self.assertFalse(report['recoveries'][0]['physics_rerun'])


if __name__=='__main__':unittest.main()
