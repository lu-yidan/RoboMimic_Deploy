"""Capture current Python/runtime provenance without claiming historical identity."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    packages = sorted((dict(name=d.metadata['Name'], version=d.version) for d in metadata.distributions()), key=lambda d: d['name'].lower())
    record = dict(python_version=sys.version, executable=sys.executable, platform=platform.platform(),
                  processor=platform.processor(), logical_processors=os.cpu_count(), torch_version=torch.__version__,
                  package_versions=packages, relevant_environment={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MUJOCO_GL')},
                  torch_threads_in_capture_process=torch.get_num_threads(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Current simulation environment capture. Per-experiment trainer explicitly sets Torch threads to one. This record does not establish package versions at earlier experiment timestamps, GPU portability, or bitwise reproduction on other machines.')
    args.output.mkdir(exist_ok=False)
    (args.output/'runtime.json').write_text(json.dumps(record, indent=2))
    (args.output/'packages.txt').write_text('\n'.join(d['name']+'=='+d['version'] for d in packages)+'\n')
    (args.output/'capture_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(output=str(args.output), python=platform.python_version(), packages=len(packages))))


if __name__ == '__main__':
    main()
