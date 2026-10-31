"""The convenient launcher must refuse stale accepted-policy bindings."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from deploy_mujoco.run_adopted_climb import verify_binding,ROOT,PACKAGE,PROOF,read


class AdoptedBindingTests(unittest.TestCase):
    def test_changed_adopted_checkpoint_or_case_cannot_launch_old_package(self):
        scratch=ROOT/'outputs/test_scratch';scratch.mkdir(exist_ok=True)
        original=read(ROOT/'research_baseline.json')
        for change in ('checkpoint','case_audit'):
            with TemporaryDirectory(dir=scratch) as temporary:
                root=Path(temporary).resolve();assert root.is_relative_to(scratch.resolve())
                baseline=deepcopy(original)
                if change=='checkpoint':baseline['policy_sha256']='0'*64
                else:baseline['multicycle_validation']['case_audits']['training_12']='0'*64
                (root/'research_baseline.json').write_text(json.dumps(baseline),encoding='utf8')
                with self.assertRaises(ValueError):verify_binding(root,PACKAGE,PROOF)


if __name__=='__main__':unittest.main()
