"""Archive only declared package files plus a completed exact-replay audit."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from deploy_mujoco.run_packaged_geometry import verify_package, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.zip.manifest.json').exists():
        raise FileExistsError(args.output)
    manifest = verify_package(args.package)
    audit_path = args.package/'replay/reproduction_audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['passed'] and audit['simulation_only']
    assert audit['release_manifest_sha256'] == sha(args.package/'release_manifest.json')
    assert set(audit['exact_arrays']) == {'qpos', 'qvel', 'time_s', 'actions', 'observations', 'torques', 'rewards'}
    assert all(audit['exact_arrays'].values()) and all(v == 0 for v in audit['maximum_differences'].values())
    members = {Path(name).as_posix():args.package/name for name in manifest['files']}
    members.update({'release_manifest.json':args.package/'release_manifest.json', 'evidence/package_reproduction_audit.json':audit_path})
    digests = {name:sha(path) for name,path in members.items()}
    for name in members:
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(members.items()):
            archive.write(path, arcname=name)
    with zipfile.ZipFile(args.output) as archive:
        assert set(archive.namelist()) == set(members)
        assert archive.testzip() is None
        for name,digest in digests.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    result = dict(archive_sha256=sha(args.output), bytes=args.output.stat().st_size, members=digests,
                  release_manifest_sha256=sha(args.package/'release_manifest.json'), reproduction_audit_sha256=sha(audit_path),
                  source_sha256=sha(__file__), scope='Declared simulation candidate package only; all ZIP members decoded and hash checked. Python interpreter and installed dependencies not bundled. Candidate selection status remains unchanged.')
    args.output.with_suffix('.zip.manifest.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(output=str(args.output), bytes=result['bytes'], members=len(members), sha256=result['archive_sha256'])))


if __name__ == '__main__':
    main()
