"""Visible raw depth and the exact normalized image consumed by the backbone."""
import time
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
        self.frame_count = 0

    def draw(self, raw, normalized, state_name, captured_at=None):
        if not self.open:
            return
        for event in pygame.event.get([pygame.QUIT]):
            self.close()
            return
        self.frame_count += 1
        self.surface.fill((20, 20, 20))
        def label(text, y):
            self.surface.blit(self.font.render(text, True, (230, 230, 230)), (12, y))
        def depth_image(values, y):
            gray = np.rint((1 - np.clip(values, 0, 1)) * 255).astype(np.uint8)
            # Keep far pixels visibly blue, rather than indistinguishable from
            # a blank/failed display. Near pixels are orange/yellow.
            t = gray.astype(np.float32) / 255
            rgb = np.stack((255*t, 60+160*t, 200*(1-t)), axis=2).astype(np.uint8)
            img = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
            self.surface.blit(pygame.transform.scale(img, (gray.shape[1]*3, gray.shape[0]*3)), (12,y))
        age = 0 if captured_at is None else max(0, (time.monotonic()-captured_at)*1000)
        label(f'{state_name} | depth age {age:.0f} ms', 8)
        if raw is None or normalized is None:
            label('No valid depth frame; check console.', 45)
        else:
            label(f'Raw: {raw.min():.2f} - {raw.max():.2f} m | near=yellow', 34)
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


def _display_worker(frames, stopped, presented):
    """SDL owns its event loop and display in this process; no MuJoCo context."""
    import queue
    import time
    preview = DepthPreview()
    payload = (None, None, 'waiting for camera', None)
    clock = pygame.time.Clock()
    while not stopped.is_set() and preview.open:
        try:
            payload = frames.get_nowait()
        except queue.Empty:
            pass
        preview.draw(*payload)
        if payload[0] is not None:
            presented.set()
        # Drain window/expose/input events here; joystick control is in the parent.
        pygame.event.get()
        clock.tick(30)
    preview.close()
    pygame.quit()


class DepthPreviewProcess:
    """Latest-frame IPC: a slow/closed preview must never stall robot control."""
    def __init__(self):
        import multiprocessing as mp
        import atexit
        context = mp.get_context('spawn')
        self.frames = context.Queue(maxsize=1)
        self.stopped = context.Event()
        self.presented = context.Event()
        self.process = context.Process(target=_display_worker, args=(self.frames, self.stopped, self.presented), daemon=True)
        self.process.start()
        self.warned = False
        atexit.register(self.close)

    @property
    def open(self):
        return self.process.is_alive() and not self.stopped.is_set()

    def draw(self, raw, normalized, state_name, captured_at=None):
        import queue
        if not self.open:
            if self.process.exitcode not in (None, 0) and not self.warned:
                print(f'[PHP depth preview] display process exited with code {self.process.exitcode}')
                self.warned = True
            return
        payload = (None if raw is None else raw.copy(),
                   None if normalized is None else normalized.copy(), state_name, captured_at)
        try:
            self.frames.put_nowait(payload)
        except queue.Full:
            pass

    def close(self):
        self.stopped.set()
        self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        self.frames.cancel_join_thread()
        self.frames.close()
