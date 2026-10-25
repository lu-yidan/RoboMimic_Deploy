"""MuJoCo/GLFW viewer with a screen-space depth inset (MuJoCo 3.2 compatible).

Uses mjr_drawPixels after rendering the 3D scene, like the WMP backend's
viewport inset. The sensor image is never projected into world geometry.
"""
import contextlib
import time
import glfw
import mujoco
import numpy as np


def depth_rgb(normalized):
    """Display-only color map; the model input is never changed."""
    values = np.asarray(normalized)
    if values.shape != (58, 87) or not np.isfinite(values).all():
        raise ValueError('Depth inset requires a finite 58x87 image')
    near = 1 - np.clip(values + .5, 0, 1)
    return np.stack((255*near, 60+160*near, 200*(1-near)), axis=-1).astype(np.uint8)


class OverlayViewer:
    """Single-threaded renderer exposing the handle used by deploy_mujoco."""
    def __init__(self, model, data, key_callback=None):
        if not glfw.init():
            raise RuntimeError('GLFW initialization failed')
        # The offscreen depth renderer leaves GLFW VISIBLE=False globally.
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        self.window = glfw.create_window(1280, 900, 'RoboMimic - MuJoCo', None, None)
        if not self.window:
            raise RuntimeError('Cannot create MuJoCo viewer window')
        self.m, self.d = model, data
        self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.user_scn = mujoco.MjvScene(model, maxgeom=10000)
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.key_callback = key_callback
        self.depth = None
        self.depth_time = 0.
        self.state_name = ''
        self.last_mouse = glfw.get_cursor_pos(self.window)
        glfw.set_key_callback(self.window, self._key)
        glfw.set_cursor_pos_callback(self.window, self._motion)
        glfw.set_mouse_button_callback(self.window, self._button)
        glfw.set_scroll_callback(self.window, self._scroll)

    def _key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif self.key_callback:
            self.key_callback(key)

    def _button(self, window, button, action, mods):
        self.last_mouse = glfw.get_cursor_pos(window)

    def _motion(self, window, x, y):
        dx, dy = x-self.last_mouse[0], y-self.last_mouse[1]
        self.last_mouse = (x, y)
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        if not (left or right or middle):
            return
        shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        height = max(1, glfw.get_window_size(window)[1])
        mujoco.mjv_moveCamera(self.m, action, dx/height, dy/height, self.scene, self.cam)

    def _scroll(self, window, x, y):
        mujoco.mjv_moveCamera(self.m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -.05*y, self.scene, self.cam)

    def set_depth(self, normalized, captured_at, state_name):
        self.depth = None if normalized is None else depth_rgb(normalized)
        self.depth_time = captured_at
        self.state_name = state_name

    def _draw_depth(self, width, height):
        scale = min(3, (width-32)//87, (height-80)//58)
        if scale < 1:
            return
        w, h = 87*scale, 58*scale
        x, y = width-w-16, height-h-56
        mujoco.mjr_rectangle(mujoco.MjrRect(x-8,y-8,w+16,h+48), .06,.06,.06,1)
        label = 'PHP depth 87x58'
        if self.depth is None:
            label += ' | no frame'
        else:
            image = np.repeat(np.repeat(self.depth,scale,axis=0),scale,axis=1)
            # MuJoCo pixels start at the lower-left, sensor arrays at the top-left.
            mujoco.mjr_drawPixels(np.ascontiguousarray(image[::-1]).ravel(), None,
                                 mujoco.MjrRect(x,y,w,h),self.context)
            age = max(0, (time.monotonic()-self.depth_time)*1000)
            label += f' | {age:.0f} ms'
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                           mujoco.MjrRect(x-8,y+h+2,w+16,36),label,'',self.context)

    def sync(self):
        glfw.make_context_current(self.window)
        width,height = glfw.get_framebuffer_size(self.window)
        if width and height:
            viewport = mujoco.MjrRect(0,0,width,height)
            mujoco.mjv_updateScene(self.m,self.d,self.opt,self.perturb,self.cam,
                                  mujoco.mjtCatBit.mjCAT_ALL,self.scene)
            # Preserve reference ghosts and policy debug spheres from the shared loop.
            for src in self.user_scn.geoms[:self.user_scn.ngeom]:
                if self.scene.ngeom >= self.scene.maxgeom:
                    break
                dst = self.scene.geoms[self.scene.ngeom]
                for field in ('type','dataid','matid','objtype','objid','category','texcoord',
                              'segid','emission','specular','shininess','reflectance','transparent','label'):
                    setattr(dst,field,getattr(src,field))
                for field in ('pos','mat','size','rgba'):
                    getattr(dst,field)[:] = getattr(src,field)
                self.scene.ngeom += 1
            mujoco.mjr_render(viewport,self.scene,self.context)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               viewport,self.state_name,'',self.context)
            self._draw_depth(width,height)
            glfw.swap_buffers(self.window)
        glfw.poll_events()

    def is_running(self):
        return self.window is not None and not glfw.window_should_close(self.window)

    def lock(self):
        return contextlib.nullcontext()

    def close(self):
        if self.window is not None:
            glfw.make_context_current(self.window)
            self.context.free()
            glfw.destroy_window(self.window)
            self.window = None

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
