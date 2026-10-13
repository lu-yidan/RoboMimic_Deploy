"""Unitree button routing sampled at control rate, independent of DDS callbacks."""
from common.remote_controller import KeyMap
from common.utils import FSMCommand
from policy.scalebfm.motion_catalog import play_selected


class RealPolicyInput:
    def __init__(self, catalog=None):
        self.catalog = catalog
        self.previous = [False] * 16

    def update(self, buttons, fsm, state, output):
        buttons = list(map(bool, buttons))
        self.previous, previous = buttons, self.previous
        # rising uses the old snapshot for this tick.
        rising = lambda key: buttons[key] and not previous[key]
        if buttons[KeyMap.F1] or buttons[KeyMap.select]:
            state.skill_cmd = FSMCommand.PASSIVE
            return
        if self.catalog is not None and not buttons[KeyMap.L1] and not buttons[KeyMap.L2]:
            delta = int(rising(KeyMap.right)) - int(rising(KeyMap.left))
            if delta:
                self.catalog.move(delta)
                print(self.catalog.describe())
        for key, command in ((KeyMap.start, FSMCommand.POS_RESET),
                             (KeyMap.B, FSMCommand.LOCO),
                             (KeyMap.A, FSMCommand.CMD_AMP)):
            if buttons[key]:
                state.skill_cmd = command
                return
        if rising(KeyMap.Y):
            state.skill_cmd = FSMCommand.CMD_SMP_RECOVERY
            return
        if self.catalog is not None and rising(KeyMap.R1):
            try:
                play_selected(self.catalog, None, None, fsm, state, output)
            except (OSError, ValueError, KeyError, AssertionError) as exc:
                print(f'[Motion] Cannot start: {exc}')
            return
        for key, command in ((KeyMap.down, FSMCommand.CMD_BEYONDMIMIC_MJ),
                             (KeyMap.up, FSMCommand.CMD_STANDUP_MJ),
                             (KeyMap.R2, FSMCommand.CMD_PINOCCHIO_1_6_MJ)):
            if buttons[key]:
                state.skill_cmd = command
                return
