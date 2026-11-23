"""Export parity must disable convolution TF32 and restore caller settings."""
import sys
from types import SimpleNamespace

import pytest

from tools.acp.export_bundle import ieee_fp32_export


def test_export_explicitly_disables_convolution_tf32_and_restores_on_error(monkeypatch):
    matmul = SimpleNamespace(fp32_precision='none')
    conv = SimpleNamespace(fp32_precision='tf32')
    cudnn = SimpleNamespace(fp32_precision='none', conv=conv)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn)))
    with pytest.raises(RuntimeError, match='failed parity'):
        with ieee_fp32_export():
            assert (matmul.fp32_precision, cudnn.fp32_precision, conv.fp32_precision) == ('ieee',)*3
            raise RuntimeError('failed parity')
    assert (matmul.fp32_precision, cudnn.fp32_precision, conv.fp32_precision) == ('none', 'none', 'tf32')


def test_export_restores_legacy_tf32_flags(monkeypatch):
    matmul, cudnn = SimpleNamespace(allow_tf32=False), SimpleNamespace(allow_tf32=True)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn)))
    with ieee_fp32_export():
        assert matmul.allow_tf32 is False and cudnn.allow_tf32 is False
    assert matmul.allow_tf32 is False and cudnn.allow_tf32 is True
