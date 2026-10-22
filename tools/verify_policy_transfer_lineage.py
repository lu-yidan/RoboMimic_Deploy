"""Verify feature implementation inheritance, including older supplemental records."""
import argparse
import inspect
import json
from pathlib import Path
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args();p=args.experiment
    m=json.loads((p/'manifest.json').read_text());parent=Path(m['policy_source']);pm=json.loads((parent/'manifest.json').read_text())
    assert sha(p/'policy.zip')==sha(parent/'policy.zip')==m['source_policy_sha256']
    model=PPO.load(p/'policy.zip',device='cpu');root=Path(__file__).resolve().parents[1]
    source=Path(inspect.getfile(type(model.policy.features_extractor))).resolve();relative=source.relative_to(root)
    inherited=next((key for key in pm['sources'] if Path(key)==relative),None)
    supplement_hash=None
    if inherited:
        expected=pm['sources'][inherited];assert sha(parent/'source_snapshot'/inherited)==expected
    else:
        record=json.loads((parent/'actor_source_provenance.json').read_text());expected=record['sha256']
        assert Path(record['source'])==relative and sha(parent/'supplemental_actor_source'/relative)==expected
        assert record['checkpoint_sha256']==sha(parent/'policy.zip') and record['audit_sha256']==sha(parent/'independent_training_audit.json')
        assert record['parent_manifest_sha256']==sha(Path(pm['policy_source'])/'manifest.json')
        supplement_hash=sha(parent/'actor_source_provenance.json')
    current=next(key for key in m['sources'] if Path(key)==relative)
    assert sha(source)==sha(p/'source_snapshot'/current)==m['sources'][current]==expected
    result=dict(passed=True,checkpoint_sha256=sha(p/'policy.zip'),parent_manifest_sha256=sha(parent/'manifest.json'),
        feature_source=str(relative),feature_sha256=expected,parent_supplement_sha256=supplement_hash,
        experiment_manifest_sha256=sha(p/'manifest.json'),source_sha256=sha(__file__),scope='Identical inherited checkpoint and feature implementation; no additional training or architecture change.')
    (p/'policy_lineage_audit.json').write_text(json.dumps(result,indent=2));(p/'policy_lineage_audit_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))


if __name__=='__main__':main()
