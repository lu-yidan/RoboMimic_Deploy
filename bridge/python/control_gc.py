"""Keep cyclic garbage collection outside the real-time ACP loop."""
from contextlib import contextmanager
import gc


@contextmanager
def pause_cyclic_gc(enabled):
    restore = enabled and gc.isenabled()
    if restore:
        gc.collect()
        gc.disable()
    try:
        yield
    finally:
        if restore:
            gc.enable()
