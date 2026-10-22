"""Keyboard input for the existing MuJoCo deployment loop (simulation only)."""
from queue import SimpleQueue
import numpy as np
from common.joystick import JoystickButton


class KeyboardJoystick:
    def __init__(self):
        self.events = SimpleQueue()
        self.paused = True
        self.velocity = np.zeros(3)
        self.released = set()
        self.hat = (0, 0)
        self.r2 = False
        self.passive = False
        self.quit = False

    def key_callback(self, key):
        self.events.put(key)

    def update(self):
        self.released.clear()
        self.hat = (0, 0)
        self.r2 = self.passive = False
        buttons = {49: JoystickButton.START, 257: JoystickButton.START,
                   50: JoystickButton.B, 51: JoystickButton.A,
                   55: JoystickButton.R1, 71: JoystickButton.L3,
                   74: JoystickButton.Y, 88: JoystickButton.X}
        while not self.events.empty():
            key = self.events.get_nowait()
            if key in buttons:
                self.released.add(buttons[key])
            elif key == 32:
                self.paused = not self.paused
                print('[Keyboard] '+('Paused' if self.paused else 'Running'), flush=True)
            elif key == 52:
                self.hat = (0, -1)
            elif key == 53:
                self.hat = (0, 1)
            elif key == 54:
                self.r2 = True
            elif key in (80, 290):  # P / F1 requests passive mode.
                self.passive = True
            elif key == 256:
                self.quit = True
            elif key == 48:
                self.velocity.fill(0)
            elif key in (87, 83, 65, 68, 81, 69):
                axis, delta = {87:(0,.1), 83:(0,-.1), 65:(1,.1),
                               68:(1,-.1), 81:(2,.1), 69:(2,-.1)}[key]
                self.velocity[axis] = np.clip(self.velocity[axis]+delta, -1., 1.)

    def is_button_pressed(self, button):
        return self.quit if button == JoystickButton.SELECT else False

    def is_button_released(self, button):
        return button in self.released

    def get_hat_direction(self):
        return self.hat

    def get_axis_value(self, axis):
        return {0:-self.velocity[1], 1:-self.velocity[0], 3:-self.velocity[2],
                2:0. if self.passive else 1., 5:1. if self.r2 else 0.}.get(axis, 0.)
