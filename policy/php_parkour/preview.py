"""Visible raw depth and the exact normalized image consumed by the backbone."""
import numpy as np
import pygame


class DepthPreview:
    def __init__(self):
        pygame.display.init()
        pygame.font.init()
        self.surface = pygame.display.set_mode((440, 460))
        pygame.display.set_caption('PHP depth: raw camera / policy input')
        self.font = pygame.font.Font(None, 22)
        self.open = True

    def draw(self, raw, normalized, state_name):
        if not self.open:
            return
        for event in pygame.event.get([pygame.QUIT]):
            self.close()
            return
        self.surface.fill((20, 20, 20))
        def label(text, y):
            self.surface.blit(self.font.render(text, True, (230, 230, 230)), (12, y))
        def depth_image(values, y):
            gray = np.rint((1 - np.clip(values, 0, 1)) * 255).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            img = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
            self.surface.blit(pygame.transform.scale(img, (gray.shape[1]*3, gray.shape[0]*3)), (12,y))
        label('Active: ' + state_name, 8)
        if raw is None or normalized is None:
            label('No valid depth frame; check console.', 45)
        else:
            label('Raw 106 x 60 | 0.3 m white -> 3 m black', 34)
            depth_image((raw - .3) / 2.7, 58)
            label('Backbone input: 87 x 58 (normalized)', 250)
            depth_image(normalized + .5, 274)
        pygame.display.flip()

    def close(self):
        if self.open:
            # JoyStick.update still needs SDL's video/event subsystem. Destroying
            # it here makes event.pump fail even though MuJoCo remains open.
            from pygame._sdl2 import Window
            Window.from_display_module().hide()
            self.open = False
