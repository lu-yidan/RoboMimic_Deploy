"""PHP/AMP button arbitration, including modifier-release ordering."""
from common.joystick import JoystickButton
from common.utils import FSMCommand


class PHPButtons:
    def __init__(self):
        self.php_a_latched = False

    def update(self, joystick, keyboard_php=False):
        command = None
        if joystick.is_button_just_pressed(JoystickButton.A) and joystick.is_button_pressed(JoystickButton.L1):
            self.php_a_latched = True
            command = FSMCommand.CMD_PHP_PARKOUR
        if joystick.is_button_released(JoystickButton.A):
            if not self.php_a_latched and not joystick.is_button_pressed(JoystickButton.L1):
                command = FSMCommand.CMD_AMP
            self.php_a_latched = False
        return FSMCommand.CMD_PHP_PARKOUR if keyboard_php else command
