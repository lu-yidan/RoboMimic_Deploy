"""Verify relocated image-only package imports and save a shareable archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    package = args.output / 'package'
    shutil.copytree(root / 'perception', package / 'perception',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(root / 'tests/fixtures/bar_perception',
                    package / 'tests/fixtures/bar_perception')
    for name in ('requirements-perception.txt', 'tests/test_bar_perception_portable.py'):
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    inputs = {str(p.relative_to(package)).replace('\\', '/'): sha(p)
              for p in package.rglob('*') if p.is_file()}
    scratch = root / 'outputs/test_scratch'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='perception relocated ', dir=scratch) as folder:
        relocated = Path(folder).resolve()
        assert relocated.is_relative_to(scratch.resolve())
        shutil.copytree(package, relocated, dirs_exist_ok=True)
        code = '''import json, pathlib, runpy, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import perception
assert pathlib.Path(perception.__file__).resolve().is_relative_to(root)
sys.argv = ['perception', '--input-dir', str(root / 'tests/fixtures/bar_perception/clean_initial'), '--output', str(root / 'prediction')]
runpy.run_module('perception', run_name='__main__')
for forbidden in ('mujoco', 'training', 'torch', 'stable_baselines3', 'g1_bar_perception'):
    assert not any(k == forbidden or k.startswith(forbidden + '.') for k in sys.modules), forbidden
actual = json.loads((root / 'prediction/prediction.json').read_text())
expected = json.loads((root / 'tests/fixtures/bar_perception/clean_initial/expected_prediction.json').read_text())
assert actual == expected
print('isolated_relocated_prediction_exact')
'''
        run = subprocess.run([sys.executable, '-I', '-c', code, str(relocated)],
                             cwd=relocated, capture_output=True, text=True)
        (args.output / 'relocation.log').write_text(run.stdout + run.stderr, encoding='utf-8')
        assert run.returncode == 0, 'Relocation failed; preserved log'
        shutil.copytree(relocated / 'prediction', args.output / 'relocated_prediction')
    archive = args.output / 'portable_bar_perception.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(inputs):
            bundle.write(package / name, name)
    report = {'passed': True, 'files': inputs, 'archive_sha256': sha(archive),
              'source_sha256': sha(Path(__file__)), 'python': sys.version,
              'scope': 'same Windows interpreter and installed dependencies; relocated path with spaces; isolated imports; saved frame only; not fresh install or Linux'}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'archive': str(archive), 'file_count': len(inputs)}))


if __name__ == '__main__':
    main()
