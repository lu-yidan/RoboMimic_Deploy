"""Exercise SDL's real background-input filter with an isolated virtual joystick."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


class JoystickBackgroundTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('xvfb-run'), 'requires Xvfb/SDL2 on Linux')
    def test_buttons_survive_depth_window_focus_loss_and_close(self):
        script = r'''
import ctypes
import glob
import pygame
from unittest.mock import patch
from common.joystick import JoyStick, JoystickButton
from policy.php_parkour.preview import DepthPreview
from pygame._sdl2 import Window

pygame.init()
libraries = glob.glob(pygame.__path__[0] + '/../pygame.libs/libSDL2-2-*.so.*')
if not libraries:
    raise SystemExit(77)
sdl = ctypes.CDLL(libraries[0])
sdl.SDL_JoystickAttachVirtual.argtypes = [ctypes.c_int] * 4
sdl.SDL_JoystickOpen.argtypes = [ctypes.c_int]
sdl.SDL_JoystickOpen.restype = ctypes.c_void_p
sdl.SDL_JoystickSetVirtualButton.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint8]
index = sdl.SDL_JoystickAttachVirtual(1, 6, 12, 1)
assert index >= 0
handle = sdl.SDL_JoystickOpen(index)
original = pygame.joystick.Joystick
with patch.object(pygame.joystick, 'Joystick', side_effect=lambda _: original(index)):
    joy = JoyStick()
preview = DepthPreview()
Window.from_display_module().hide()
for _ in range(5):
    pygame.event.pump()
    pygame.time.wait(10)
assert not pygame.key.get_focused()
sdl.SDL_JoystickSetVirtualButton(handle, int(JoystickButton.START), 1)
joy.update()
assert joy.is_button_just_pressed(JoystickButton.START)
sdl.SDL_JoystickSetVirtualButton(handle, int(JoystickButton.START), 0)
joy.update()
assert joy.is_button_released(JoystickButton.START)
preview.close()
sdl.SDL_JoystickSetVirtualButton(handle, int(JoystickButton.A), 1)
sdl.SDL_JoystickSetVirtualButton(handle, int(JoystickButton.L1), 1)
joy.update()
assert joy.is_button_pressed(JoystickButton.A)
assert joy.is_button_pressed(JoystickButton.L1)
pygame.quit()
'''
        env = dict(os.environ, SDL_AUDIODRIVER='dummy', SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS='0')
        env.pop('SDL_VIDEODRIVER', None)
        result = subprocess.run(['xvfb-run', '-a', sys.executable, '-c', script],
                                cwd=str(Path(__file__).resolve().parents[1]), env=env,
                                capture_output=True, text=True, timeout=30)
        if result.returncode == 77:
            self.skipTest('SDL shared library not available for virtual-device test')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
