#!/usr/bin/env python3
"""Build the native viewer-only depth backport for Linux / MuJoCo 3.2.3.

Run in the deployment Python environment. --restore reinstates the saved viewer.
The libmujoco physics engine and other Python extensions are never replaced.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request

VERSION = '3.2.3'
SOURCE_SHA = 'dd617f40c1c044f5df7fbc9333d3094d3c88629d4e3d86224e5cb42de40a6a8b'
DEPS = [
    ('abseil-cpp', 'abseil/abseil-cpp', '4447c7562e3bc702ade25105912dce503f0c4010',
     'df8b3e0da03567badd9440377810c39a38ab3346fa89df077bb52e68e4d61e74'),
    ('pybind11', 'pybind/pybind11', 'a2e59f0e7065404b44dfe92a28aca47ba1378dc4',
     '1da0a5c38950e4f49329e9de10c59b2527840a1b963113f1c53b455971cd6aef'),
    ('glfw3', 'glfw/glfw', '7482de6071d21db77a7236155da44c172a7f6c9e',
     'b59ec233ee8435831511cda07c83a5f3f49225db390a7d30201160483d920fcb'),
    ('lodepng', 'lvandeve/lodepng', 'b4ed2cd7ecf61d29076169b49199371456d4f90b',
     'b67e466ba659c07ac775d07dbd97af319cde449ce14abed9ae596df29d888603'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, path, expected):
    if not path.exists() or sha(path) != expected:
        temporary = path.with_suffix('.download')
        with urllib.request.urlopen(url, timeout=120) as response:
            temporary.write_bytes(response.read())
        if sha(temporary) != expected:
            raise RuntimeError('Checksum mismatch: ' + url)
        temporary.replace(path)


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive)) as source:
        for item in source.getmembers():
            target = (destination / item.name).resolve()
            if destination.resolve() not in target.parents or item.issym() or item.islnk():
                raise RuntimeError('Unsafe archive member: ' + item.name)
        source.extractall(str(destination))
        return destination / source.getnames()[0].split('/')[0]


def replace(source, target):
    temporary = target.with_name(target.name + '.installing')
    shutil.copy2(str(source), str(temporary))
    os.replace(str(temporary), str(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    import mujoco
    from mujoco import _simulate
    if sys.platform != 'linux' or mujoco.__version__ != VERSION:
        parser.error('This backport supports Linux with mujoco==3.2.3 only.')
    installed = Path(_simulate.__file__)
    backup = installed.with_name(installed.name + '.before-depth')
    record = installed.with_name('native-depth-overlay.json')
    if args.restore:
        if not backup.exists():
            parser.error('No original viewer backup found: ' + str(backup))
        replace(backup, installed)
        if record.exists():
            record.unlink()
        print('Original native viewer restored. Restart the simulation.')
        return
    patch = Path(__file__).resolve().parent / 'native_viewer/mujoco-3.2.3-depth.patch'
    patch_sha = sha(patch)
    installer_sha = sha(Path(__file__))
    if record.exists() and hasattr(_simulate.Simulate, 'set_depth_overlay'):
        metadata = json.loads(record.read_text())
        if metadata.get('patch_sha256') == patch_sha and metadata.get('installer_sha256') == installer_sha and metadata.get('extension_sha256') == sha(installed):
            print('Native depth overlay is already installed.')
            return
    compiler = os.environ.get('CXX') or shutil.which('clang++-18') or shutil.which('clang++')
    if not compiler:
        parser.error('Clang/libc++ is required to match the official MuJoCo wheel; see docs/native_depth_overlay.md')
    for command in ('cmake', 'patch'):
        if not shutil.which(command):
            parser.error('Missing build prerequisite: ' + command + '; see docs/native_depth_overlay.md')
    if not shutil.which('xvfb-run') and not os.environ.get('DISPLAY'):
        parser.error('Install xvfb and xauth for the native window verification.')
    cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache'))) / 'robomimic-native-viewer'
    cache.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256((str(installed)+compiler).encode()).hexdigest()[:12]
    work = cache / ('build-' + identity + '-' + patch_sha[:12])
    work.mkdir(exist_ok=True)
    source_archive = cache / ('mujoco-' + VERSION + '.tar.gz')
    if not source_archive.exists() or sha(source_archive) != SOURCE_SHA:
        with urllib.request.urlopen('https://pypi.org/pypi/mujoco/'+VERSION+'/json', timeout=60) as response:
            metadata = json.load(response)
        url = next(x['url'] for x in metadata['urls'] if x['packagetype'] == 'sdist')
        download(url, source_archive, SOURCE_SHA)
    source = work / ('mujoco-' + VERSION)
    if not source.exists():
        source = extract(source_archive, work)
        subprocess.run(['patch', '-p1', '-i', str(patch)], cwd=str(source), check=True)
    cmake_file = source / 'mujoco/CMakeLists.txt'
    content = cmake_file.read_text()
    if 'find_package(Eigen3 REQUIRED)' not in content:
        start = content.index('findorfetch(', content.index('# ==================== EIGEN'))
        end = content.index('\n)', start) + 2
        cmake_file.write_text(content[:start] + 'find_package(Eigen3 REQUIRED)' + content[end:])
    deps = cache / 'deps'
    deps.mkdir(exist_ok=True)
    def fetch(item):
        name, repo, revision, digest = item
        archive = deps / (name + '.tar.gz')
        download('https://codeload.github.com/'+repo+'/tar.gz/'+revision, archive, digest)
        # Build cache is scoped by patch and environment identity.
        destination = work / ('dep-' + name)
        if destination.exists():
            return name, next(destination.iterdir())
        return name, extract(archive, destination)
    with ThreadPoolExecutor(max_workers=4) as executor:
        dependencies = list(executor.map(fetch, DEPS))
    package = Path(mujoco.__file__).parent
    physics = package / ('libmujoco.so.' + VERSION)
    physics_sha = sha(physics)
    build = work / 'cmake-build'
    command = ['cmake', '-S', str(source/'mujoco'), '-B', str(build),
               '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=OFF',
               '-DCMAKE_CXX_COMPILER='+compiler,
               '-DCMAKE_MODULE_LINKER_FLAGS=-static-libstdc++ -Wl,--exclude-libs,ALL -Wl,--whole-archive -l:libunwind.a -Wl,--no-whole-archive '+os.environ.get('LDFLAGS', ''),
               '-DCMAKE_CXX_FLAGS=-stdlib=libc++ -Wno-unused-command-line-argument '+os.environ.get('CXXFLAGS', ''),
               '-DCMAKE_MODULE_PATH='+str(source/'mujoco/cmake'),
               '-DMUJOCO_LIBRARY_DIR='+str(package), '-DMUJOCO_INCLUDE_DIR='+str(package/'include'),
               '-DMUJOCO_LIBRARY='+str(physics), '-DMUJOCO_INCLUDE='+str(package/'include'),
               '-DPython3_ROOT_DIR='+sys.prefix, '-DPython3_EXECUTABLE='+sys.executable,
               '-DPYTHON_EXECUTABLE='+sys.executable,
               '-DPYTHON_INCLUDE_DIR='+sysconfig.get_path('include'),
               '-DPYTHON_LIBRARY='+str(Path(sysconfig.get_config_var('LIBDIR'))/sysconfig.get_config_var('LDLIBRARY'))]
    command += ['-DFETCHCONTENT_SOURCE_DIR_'+name.upper()+'='+str(path) for name,path in dependencies]
    subprocess.run(command, check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', '_simulate', '-j', '4'], check=True)
    extension = build / installed.name
    if not extension.exists():
        raise RuntimeError('Build produced the wrong Python ABI; no files installed.')
    if not backup.exists():
        shutil.copy2(str(installed), str(backup))
    previous = work / installed.name
    shutil.copy2(str(installed), str(previous))
    try:
        replace(extension, installed)
        probe = [sys.executable, '-c',
                 'import faulthandler; faulthandler.enable(); import time; import mujoco, mujoco.viewer, numpy as np; '
                 'm=mujoco.MjModel.from_xml_string("<mujoco/>"); d=mujoco.MjData(m); list(d.contact); '
                 'v=mujoco.viewer.launch_passive(m,d); '
                 'v._sim().set_depth_overlay(np.zeros((58,87,3),np.uint8),"test"); '
                 'v.sync(); time.sleep(.5); v.close(); time.sleep(.5)']
        if shutil.which('xvfb-run'):
            probe = ['xvfb-run', '-a'] + probe
        subprocess.run(probe, check=True, timeout=30)

        if sha(physics) != physics_sha:
            raise RuntimeError('Physics library checksum changed unexpectedly')
    except Exception:
        replace(previous, installed)
        raise
    record.write_text(json.dumps({'patch_sha256':patch_sha, 'installer_sha256':installer_sha, 'extension_sha256':sha(installed),
                                  'physics_sha256':physics_sha, 'backup':str(backup)}, indent=2)+'\n')
    print('Installed native depth overlay. Physics library unchanged. Restart the simulation.')


if __name__ == '__main__':
    main()
