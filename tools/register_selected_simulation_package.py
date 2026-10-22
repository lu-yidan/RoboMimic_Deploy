"""Attach completed selection evidence without rewriting an immutable candidate."""
import argparse
import json
from pathlib import Path
from deploy_mujoco.run_packaged_geometry import verify_package, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selection = json.loads(args.selection.read_text())
    for path, digest in selection['evidence'].items():
        assert sha(path) == digest
    manifest = verify_package(args.package)
    replay = json.loads((args.package/'replay/reproduction_audit.json').read_text())
    archive = json.loads(args.archive.with_suffix('.zip.manifest.json').read_text())
    assert replay['passed'] and replay['release_manifest_sha256'] == sha(args.package/'release_manifest.json')
    assert archive['archive_sha256'] == sha(args.archive)
    assert archive['release_manifest_sha256'] == sha(args.package/'release_manifest.json')
    assert archive['reproduction_audit_sha256'] == sha(args.package/'replay/reproduction_audit.json')
    matching = selection['checkpoint_sha256'] == sha(args.package/manifest['checkpoint'])
    result = dict(accepted_as_selected_simulation_package=matching,
                  reason='Selected checkpoint and exact simulation replay match' if matching else 'Candidate package is not the selected checkpoint; retain as candidate only',
                  selection_sha256=sha(args.selection), selected_experiment=selection['selected_experiment'],
                  selected_checkpoint_sha256=selection['checkpoint_sha256'], package_checkpoint_sha256=sha(args.package/manifest['checkpoint']),
                  archive_sha256=sha(args.archive), archive_path=str(args.archive.resolve()),
                  package_manifest_sha256=sha(args.package/'release_manifest.json'), demonstrated_case=manifest['case'],
                  source_sha256=sha(__file__), scope='Post-selection receipt only. Original candidate archive remains byte-identical. One packaged demonstration; inherited model-selection validation is not an independent test or hardware certification.')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
