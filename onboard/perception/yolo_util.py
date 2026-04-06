"""Select YOLO weights: TensorRT .engine only when ``import tensorrt`` works (Jetson-safe)."""

from __future__ import annotations

import os
import sys


def _ensure_jetson_trt_env():
    """Patch env so ``import tensorrt`` works on Jetson even without run_fused.sh."""
    cuda_compat = "/usr/local/cuda-12.1/compat"
    if os.path.isdir(cuda_compat):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if cuda_compat not in ld:
            os.environ["LD_LIBRARY_PATH"] = f"{cuda_compat}:{ld}" if ld else cuda_compat
    sys_dist = "/usr/lib/python3.8/dist-packages"
    if os.path.isdir(sys_dist) and sys_dist not in sys.path:
        sys.path.insert(0, sys_dist)


def tensorrt_python_ok() -> bool:
    _ensure_jetson_trt_env()
    try:
        import tensorrt  # noqa: F401
        return True
    except Exception:
        return False


def resolve_model_path(model_path: str) -> str:
    """Prefer sibling ``.engine`` for a ``.pt`` path only if TensorRT Python bindings exist.

    Avoids Ultralytics trying ``pip install tensorrt`` on Jetson/aarch64 when conda has no TRT.
    """
    if model_path.endswith('.pt'):
        engine_path = model_path.replace('.pt', '.engine')
        if os.path.isfile(engine_path) and tensorrt_python_ok():
            print(f"[INFO] TensorRT engine found: {engine_path}")
            return engine_path
        if os.path.isfile(engine_path):
            print(
                f"[WARN] {engine_path} exists but `import tensorrt` failed — "
                f"using .pt (Jetson: PYTHONPATH may need /usr/lib/python3.8/dist-packages)."
            )
        else:
            print(f"[INFO] No .engine at {engine_path}, using .pt")
        return model_path
    if model_path.endswith('.engine') and not tensorrt_python_ok():
        pt_path = model_path.replace('.engine', '.pt')
        if os.path.isfile(pt_path):
            print(
                f"[WARN] TensorRT Python unavailable; using {pt_path} instead of {model_path}"
            )
            return pt_path
        raise RuntimeError(
            f"Model {model_path!r} requires importable tensorrt; no fallback at {pt_path!r}."
        )
    return model_path
