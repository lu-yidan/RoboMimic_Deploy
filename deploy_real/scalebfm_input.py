"""Unitree button routing sampled at control rate, independent of DDS callbacks."""
from common.remote_controller import KeyMap
from common.utils import FSMCommand, FSMStateName
from policy.scalebfm.motion_catalog import play_selected


class RealPolicyInput:
    def __init__(self, catalog=None, php_enabled=False):
        self.catalog = catalog
        self.previous = [False] * 16
        self.php_enabled = php_enabled
        self.php_a_latched = False
        self.php_chord = False

    def update(self, buttons, fsm, state, output):
        buttons = list(map(bool, buttons))
        self.previous, previous = buttons, self.previous
        # rising uses the old snapshot for this tick.
        rising = lambda key: buttons[key] and not previous[key]
        chord = self.php_enabled and buttons[KeyMap.L1] and buttons[KeyMap.A]
        php_edge = chord and not self.php_chord
        self.php_chord = chord
        if chord:
            self.php_a_latched = True
        a_released = previous[KeyMap.A] and not buttons[KeyMap.A]
        amp_edge = a_released and not self.php_a_latched and not buttons[KeyMap.L1]
        if a_released:
            self.php_a_latched = False
        if buttons[KeyMap.F1] or buttons[KeyMap.select]:
            state.skill_cmd = FSMCommand.PASSIVE
            return
        # Recovery and damping preempt PHP, including a simultaneous chord.
        if rising(KeyMap.Y) and not buttons[KeyMap.L1]:
            state.skill_cmd = FSMCommand.CMD_SMP_RECOVERY
            return
        if php_edge:
            state.skill_cmd = FSMCommand.CMD_PHP_PARKOUR
            return
        # Unitree's 16-bit remote packet has no R3. F2 replaces sim's R3.
        if self.php_enabled and rising(KeyMap.F2) and fsm.cur_policy.name == FSMStateName.SKILL_PHP_PARKOUR:
            state.php_high_speed = not state.php_high_speed
        if self.catalog is not None and not buttons[KeyMap.L1] and not buttons[KeyMap.L2]:
            delta = int(rising(KeyMap.right)) - int(rising(KeyMap.left))
            if delta:
                self.catalog.move(delta)
                print(self.catalog.describe())
        for key, command in ((KeyMap.start, FSMCommand.POS_RESET),
                             (KeyMap.B, FSMCommand.LOCO)):
            if buttons[key]:
                state.skill_cmd = command
                return
        if (amp_edge if self.php_enabled else buttons[KeyMap.A]):
            state.skill_cmd = FSMCommand.CMD_AMP
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
