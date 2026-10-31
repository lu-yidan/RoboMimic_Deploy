"""Complete reporting for the unchanged fresh baseline after early-failure audit repair."""
import argparse
import json
from pathlib import Path
from tools.compare_audited_closure_candidates import load_run, sha
from tools.audit_trained_closure_controller_v2 import audit as audit_fixed
from tools.audit_closure_foreaft_reference import audit as audit_added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = load_run(args.run)
    manifest = json.loads((args.run/'manifest.json').read_text())
    protocol = json.loads((args.run/'holdout_protocol.json').read_text())
    assert manifest['fresh_role'] == 'baseline'
    assert manifest['holdout_protocol_sha256'] == sha(args.run/'holdout_protocol.json')
    assert run['cases'] == protocol['cases']
    selected = protocol['candidates']['baseline']
    assert run['checkpoint_sha256'] == selected['checkpoint_sha256']
    assert run['adaptive_reference_command'] == selected['command']
    original = sha(args.run/'manifest.json'), sha(args.run/'summary.json')
    audit_fixed(args.run)
    audit_added(args.run)
    assert original == (sha(args.run/'manifest.json'), sha(args.run/'summary.json'))
    result = dict(successes=sum(row['success'] for row in run['rows'].values()),
                  cases=len(run['rows']), protocol_sha256=manifest['holdout_protocol_sha256'],
                  automatic_promotion=False, strata={})
    for group in sorted(set(protocol['strata'].values())):
        keys = [key for key, value in protocol['strata'].items() if value == group]
        result['strata'][group] = dict(cases=len(keys), successes=sum(run['rows'][key]['success'] for key in keys))
    destination = args.run/'geometry_result.json'
    assert not destination.exists()
    destination.write_text(json.dumps(result, indent=2))
    evidence = dict(manifest_and_summary_unchanged=True, original_failed_audit_sha256=sha(args.run/'controller_modifier_audit.json'),
                    corrected_audit_sha256=sha(args.run/'controller_modifier_audit_v2.json'), source_sha256=sha(__file__),
                    scope='Reporting-only repair: early failure before closure requires no waist trace. No new simulation, policy change or success relabeling.')
    (args.run/'report_repair.json').write_text(json.dumps(evidence, indent=2))
    (args.run/'report_repair_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
