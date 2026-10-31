"""Explicit, process-local authority intervention; frozen training sources stay intact."""
import inspect
import math
import sys
import textwrap


def install(authority):
    from training.full_cycle_residual_env import FullCycleResidualEnv
    cls = FullCycleResidualEnv
    if not math.isfinite(authority) or not 0 < authority <= 0.03:
        raise ValueError('Invalid residual authority')
    if getattr(cls, '_explicit_authority', None) == authority:
        return
    for name, old, new in (
        ('step', '.015*action', f'({authority!r})*action'),
        ('observation', 'self.residual/.015', f'self.residual/({authority!r})'),
    ):
        source = textwrap.dedent(inspect.getsource(getattr(cls, name)))
        assert source.count(old) == 1, (name, 'Unexpected source or another active override')
        namespace = dict(vars(sys.modules[cls.__module__]))
        exec(compile(source.replace(old, new), __file__ + ':' + name, 'exec'), namespace)
        setattr(cls, name, namespace[name])
    cls._explicit_authority = authority
