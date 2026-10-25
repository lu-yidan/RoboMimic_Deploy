"""Download pinned official ScaleBFM source and M weights outside this repository."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

REVISION = 'abd6f17c02fe0baabc14709feb8d9ea4959aa621'
SOURCE = 'https://github.com/zengweishuai/ScaleBFM.git'
BASE = 'https://huggingface.co/WeishuaiZeng/ScaleBFM/resolve/main/'
FILES = {
    'checkpoint/humanoid_transformer_m/model_22200.pt': '88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3',
    'compiled_checkpoint/humanoid_transformer_m/linux/model_22200_tensorrt_metadata.json': '3024f18136314fade2676e22d6eafb1a8b2befa388da55b0885a53fcb1368d16',
    'compiled_checkpoint/humanoid_transformer_m/linux/mode_table.pt': 'fca11f2399bced32517b51c7f9dcf13c97117ca469b0b32e207d8f7d664fbde9',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3]/'ScaleBFM')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    if root == repo or repo in root.parents:
        parser.error('Choose an external directory so model downloads stay out of the deployment repository')
    if not root.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse', SOURCE, str(root)], check=True)
        subprocess.run(['git', '-C', str(root), 'fetch', '--depth', '1', 'origin', REVISION], check=True)
        subprocess.run(['git', '-C', str(root), 'checkout', '--detach', REVISION], check=True)
        subprocess.run(['git', '-C', str(root), 'sparse-checkout', 'set', 'ScaleBridge', 'ScaleTrack'], check=True)
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        raise ValueError(f'Existing source revision {actual} differs from validated {REVISION}; use a new --root directory')
    exporter = root/'ScaleTrack/scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer_onboard.py'
    if not exporter.is_file():
        raise FileNotFoundError('ScaleTrack sources missing; include ScaleBridge and ScaleTrack in your sparse checkout')
    target = root/'artifacts/scalebfm/humanoid_transformer_m'
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for remote, expected in FILES.items():
        path = target/Path(remote).name
        if path.exists():
            if digest(path) != expected:
                raise ValueError(f'Existing file checksum mismatch: {path}; not overwriting')
        else:
            partial = path.with_suffix(path.suffix+'.part')
            print(f'Downloading {remote}', flush=True)
            with urllib.request.urlopen(BASE+remote, timeout=60) as source, partial.open('wb') as dest:
                for chunk in iter(lambda: source.read(1024*1024), b''):
                    dest.write(chunk)
            if digest(partial) != expected:
                raise ValueError(f'Download checksum mismatch: {partial}')
            partial.rename(path)
        records.append(dict(source=BASE+remote, file=path.name, bytes=path.stat().st_size, sha256=expected))
    (target/'download_manifest.json').write_text(json.dumps(dict(source_revision=REVISION, files=records), indent=2)+'\n')
    print(f'Ready: {target}\nSet scalebfm.root to {root} in config/deploy.yaml')


if __name__ == '__main__':
    main()
