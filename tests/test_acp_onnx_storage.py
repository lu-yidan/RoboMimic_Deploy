"""The bundle policy hash must cover the parameters used by inference."""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import pytest

from tools.acp.export_bundle import inline_onnx_parameters
from policy.acp.onnx_storage import require_self_contained_model


def external_model(directory):
    graph = helper.make_graph([helper.make_node('Add', ['input', 'weight'], ['output'])], 'external_weights',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(np.array([[2., 3.]], dtype=np.float32), name='weight')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=9)
    path = directory / 'policy.onnx'
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location='weights.bin', size_threshold=0)
    return path


def test_verified_bytes_reject_external_weight_dependency(tmp_path):
    path = external_model(tmp_path)
    # A matching protobuf hash alone cannot authenticate weights.bin.
    assert (tmp_path / 'weights.bin').is_file()
    with pytest.raises(ValueError, match='self-contained'):
        require_self_contained_model(path.read_bytes())


def test_exported_policy_is_portable_and_contains_its_parameters(tmp_path):
    path = external_model(tmp_path)
    inline_onnx_parameters(path)
    # Copy only the authenticated model into a directory without weights.bin.
    destination = tmp_path / 'portable'
    destination.mkdir()
    copied = destination / 'policy.onnx'
    copied.write_bytes(path.read_bytes())
    require_self_contained_model(copied.read_bytes())
    session = ort.InferenceSession(copied.read_bytes(), providers=['CPUExecutionProvider'])
    actual = session.run(None, {'input': np.array([[1., 1.]], dtype=np.float32)})[0]
    np.testing.assert_array_equal(actual, [[3., 4.]])
