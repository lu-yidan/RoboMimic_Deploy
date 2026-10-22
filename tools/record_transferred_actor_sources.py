"""Supplement an immutable evaluation with its inherited custom actor source."""
import inspect
import json
from pathlib import Path
import shutil
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import sha


def main():
    p=Path('outputs/highrod_existing_oracle_policy_v1');m=json.loads((p/'manifest.json').read_text());a=json.loads((p/'independent_training_audit.json').read_text())
    parent=Path(m['policy_source']);pm=json.loads((parent/'manifest.json').read_text())
    assert a['passed'] and a['checkpoint_sha256']==sha(p/'policy.zip')==sha(parent/'policy.zip')
    model=PPO.load(p/'policy.zip',device='cpu')
    root=Path(__file__).resolve().parents[1];source=Path(inspect.getfile(type(model.policy.features_extractor))).resolve();relative=source.relative_to(root)
    key=next(k for k in pm['sources'] if Path(k)==relative)
    inherited=parent/'source_snapshot'/relative
    assert sha(inherited)==sha(source)==pm['sources'][key]
    target=p/'supplemental_actor_source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(inherited,target)
    result=dict(custom_features_class=type(model.policy.features_extractor).__module__+'.'+type(model.policy.features_extractor).__name__,
        source=str(relative),sha256=sha(target),copied_from=str(inherited),checkpoint_sha256=sha(p/'policy.zip'),
        audit_sha256=sha(p/'independent_training_audit.json'),parent_manifest_sha256=sha(parent/'manifest.json'),
        source_record_sha256=sha(__file__),scope='Supplement to unchanged completed evaluation manifest. Custom feature module was imported by checkpoint loading after initial source inventory. It matches the frozen parent and current replay implementation exactly; existing audit already reproduces every action. No rerun or result relabelling.')
    (p/'actor_source_provenance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
