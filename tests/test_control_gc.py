import gc
import unittest
from unittest.mock import patch

from bridge.python.control_gc import pause_cyclic_gc


class ControlGCTest(unittest.TestCase):
    def test_collects_before_loop_and_restores_after_exception(self):
        with patch.object(gc, 'isenabled', return_value=True), \
                patch.object(gc, 'collect') as collect, \
                patch.object(gc, 'disable') as disable, \
                patch.object(gc, 'enable') as enable:
            with self.assertRaisesRegex(RuntimeError, 'loop failed'):
                with pause_cyclic_gc(True):
                    collect.assert_called_once_with()
                    disable.assert_called_once_with()
                    enable.assert_not_called()
                    raise RuntimeError('loop failed')
            enable.assert_called_once_with()

    def test_preserves_disabled_gc_and_other_policy_behavior(self):
        for requested, initial in [(True, False), (False, True)]:
            with self.subTest(requested=requested, initial=initial):
                with patch.object(gc, 'isenabled', return_value=initial), \
                        patch.object(gc, 'collect') as collect, \
                        patch.object(gc, 'disable') as disable, \
                        patch.object(gc, 'enable') as enable:
                    with pause_cyclic_gc(requested):
                        pass
                    collect.assert_not_called()
                    disable.assert_not_called()
                    enable.assert_not_called()
