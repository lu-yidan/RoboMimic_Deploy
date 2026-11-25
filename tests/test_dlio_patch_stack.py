"""Upgrading overlapping vendor patches must preserve edits and roll back errors."""
import difflib
from pathlib import Path
import subprocess

import pytest

from onboard.localization.dlio.prepare_sources import apply_patch_stack


@pytest.mark.parametrize('initial', ['clean', 'base', 'complete', 'unrelated_edit'])
def test_patch_stack_upgrade_is_repeatable_and_preserves_unrelated_edits(tmp_path, initial):
    repo = tmp_path/'source'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    original = 'zero\none\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\n'
    base = original.replace('two', 'base-two')
    complete = base.replace('three', 'runtime-three')
    patches = []
    for name, before, after in [('base', original, base), ('runtime', base, complete)]:
        path = tmp_path/(name+'.patch')
        path.write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                    fromfile='a/value.txt', tofile='b/value.txt')))
        patches.append(path)
    path = repo/'value.txt'
    path.write_text({'clean': original, 'base': base, 'complete': complete,
                     'unrelated_edit': base.replace('nine', 'my-nine')}[initial])
    expected = complete.replace('nine', 'my-nine') if initial == 'unrelated_edit' else complete
    apply_patch_stack(repo, patches)
    assert path.read_text() == expected
    apply_patch_stack(repo, patches)
    assert path.read_text() == expected


def test_patch_stack_restores_original_bytes_when_later_patch_cannot_apply(tmp_path):
    repo = tmp_path/'source'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    original = 'a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n'
    base = original.replace('b\n', 'base-b\n')
    # The second patch targets a different source version.
    patches = []
    for name, before, after in [('base', original, base), ('runtime', 'alien\n', 'runtime\n')]:
        path = tmp_path/(name+'.patch')
        path.write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                    fromfile='a/value.txt', tofile='b/value.txt')))
        patches.append(path)
    path = repo/'value.txt'
    saved = base.replace('j\n', 'user-j\n').encode()
    path.write_bytes(saved)
    with pytest.raises(subprocess.CalledProcessError):
        apply_patch_stack(repo, patches)
    assert path.read_bytes() == saved
