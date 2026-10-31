"""Keep the ordinary physical/policy audit and bind the new-reference protocol."""
import argparse
from pathlib import Path
from tools.audit_compiled_hook_pair import audit as audit_compiled
from tools.validate_swing_reference import read, write, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    out=p.parse_args().output.resolve()
    manifest=read(out/'manifest.json')
    protocol=Path(manifest['reference_retarget_protocol'])
    assert manifest['experimental_reference'] and not manifest['automatic_promotion']
    assert sha(protocol)==manifest['reference_retarget_protocol_sha256']
    declared=read(protocol)
    assert manifest['case']==declared['case'] and manifest['arm']==declared['arm']=='adopted_v2'
    assert manifest['plan_sha256']==declared['plan_sha256']
    assert sha(Path(manifest['plan_path']).parent/'manifest.json')==declared['reference_manifest_sha256']
    assert sha(Path(manifest['plan_path']).parent/'audit.json')==declared['reference_audit_sha256']
    audit=audit_compiled(out)
    write(out/'reference_retarget_audit.json',dict(passed=audit['passed'],
        compiled_audit_sha256=sha(out/'compiled_pair_audit.json'),
        protocol_sha256=sha(protocol),manifest_sha256=sha(out/'manifest.json'),
        source_sha256=sha(__file__),scope='Original physical/policy/integrity criteria plus explicit experimental-reference binding; no baseline promotion.'))


if __name__=='__main__':main()
