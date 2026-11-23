"""Bundle schema 1 authenticates one self-contained ONNX file."""
import onnx


def require_self_contained_model(model_bytes):
    """Reject external tensors anywhere, including nested graphs/functions."""
    def visit(message):
        if isinstance(message, onnx.TensorProto):
            if message.data_location == onnx.TensorProto.EXTERNAL or message.external_data:
                raise ValueError('ACP policy must be self-contained ONNX; export the bundle again')
        for field, value in message.ListFields():
            if field.type == field.TYPE_MESSAGE:
                repeated = getattr(field, 'is_repeated', None)
                if repeated is None:  # Protobuf versions used by Python 3.8.
                    repeated = field.label == field.LABEL_REPEATED
                if repeated:
                    for child in value:
                        visit(child)
                else:
                    visit(value)
    visit(onnx.load_model_from_string(model_bytes))
