# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Compatibility layer for the legacy PyParticles OpenGL animation.

The historical renderer is kept intact.  This module adapts the pieces whose
runtime behaviour changed between the 2012 PyOpenGL/GLUT stack and current
FreeGLUT/PyOpenGL/NumPy releases.  It also provides lifecycle hooks needed to
create OpenCL/OpenGL shared resources after the GL context exists.
"""

import ctypes
import signal

import pyparticles.animation.animated_ogl as legacy
import pyparticles.ogl.draw_particles_ogl as legacy_draw
import pyparticles.ogl.draw_particles_ogl_compat as draw_compat
import pyparticles.ogl.draw_vector_field as legacy_vector_field


legacy_draw.DrawParticlesGL = draw_compat.DrawParticlesGL
legacy_draw.charged_particles_color = draw_compat.charged_particles_color
legacy_draw.charged_particles_vect_color = draw_compat.charged_particles_vect_color
legacy.drp.DrawParticlesGL = draw_compat.DrawParticlesGL


def _safe_vector_field_del(self):
    pass


legacy_vector_field.DrawVectorField.__del__ = _safe_vector_field_del


_legacy_key_pressed = legacy.KeyPressed


def _leave_main_loop():
    """Request a clean FreeGLUT exit when the extension is available."""
    leave = getattr(legacy, "glutLeaveMainLoop", None)
    if leave is not None:
        try:
            if bool(leave):
                leave()
                return
        except Exception:
            pass

    raise KeyboardInterrupt


def _key_pressed(key, x, y):
    """Adapt modern PyOpenGL byte keyboard callbacks to the legacy handler."""
    if isinstance(key, bytes):
        key = key.decode("latin-1")

    if key in ("q", "Q", "\x1b"):
        _leave_main_loop()
        return

    return _legacy_key_pressed(key, x, y)


def _sigint_handler(signum, frame):
    _leave_main_loop()


def _close_window():
    _leave_main_loop()


def _configure_freeglut_exit():
    """Ask FreeGLUT to return from its main loop when a window is closed."""
    set_option = getattr(legacy, "glutSetOption", None)
    action_key = getattr(legacy, "GLUT_ACTION_ON_WINDOW_CLOSE", None)
    return_action = getattr(legacy, "GLUT_ACTION_GLUTMAINLOOP_RETURNS", None)

    if set_option is not None and action_key is not None and return_action is not None:
        try:
            if bool(set_option):
                set_option(action_key, return_action)
        except Exception:
            pass

    close_func = getattr(legacy, "glutCloseFunc", None)
    if close_func is not None:
        try:
            if bool(close_func):
                close_func(_close_window)
        except Exception:
            pass


class AnimatedGl(legacy.AnimatedGl):
    """Legacy renderer with modern FreeGLUT and CL/GL lifecycle hooks."""

    def __init__(self):
        super(AnimatedGl, self).__init__()
        self.__gl_context_ready_callback = None
        self.__post_step_callback = None
        self.__cleanup_callbacks = []

    def get_trajectory_step(self):
        return legacy.pan.Animation.get_trajectory_step(self)

    def set_trajectory_step(self, value):
        legacy.pan.Animation.set_trajectory_step(self, value)
        self.draw_particles.set_trajectory_step(value)

    trajectory_step = property(get_trajectory_step, set_trajectory_step)

    def set_gl_context_ready_callback(self, callback):
        """Run *callback(animation)* immediately after GLUT creates the window.

        At that point the GL context is current, but the legacy builder has not
        yet called ``ode_solver.update_force()``.  A callback may therefore
        create a GL-sharing OpenCL context and replace ``animation.ode_solver``
        before the first force evaluation.
        """
        self.__gl_context_ready_callback = callback

    def set_post_step_callback(self, callback):
        """Run *callback(animation)* after each solver step and before draw."""
        self.__post_step_callback = callback

    def add_cleanup_callback(self, callback):
        self.__cleanup_callbacks.append(callback)

    def build_animation(self):
        legacy.ctypes = ctypes
        legacy.KeyPressed = _key_pressed

        # The legacy builder creates the GLUT window and immediately evaluates
        # the force.  Wrap just the window creation call so CL/GL resources can
        # be established in the narrow interval where the GL context is current
        # and before the solver is first used.
        original_create_window = legacy.glutCreateWindow

        def create_window_with_hook(*args, **kwargs):
            window = original_create_window(*args, **kwargs)
            callback = self.__gl_context_ready_callback
            if callback is not None:
                callback(self)
            return window

        legacy.glutCreateWindow = create_window_with_hook
        try:
            result = super(AnimatedGl, self).build_animation()
        finally:
            legacy.glutCreateWindow = original_create_window

        _configure_freeglut_exit()
        return result

    def data_stream(self):
        result = super(AnimatedGl, self).data_stream()
        callback = self.__post_step_callback
        if callback is not None:
            callback(self)
        return result

    def start(self):
        previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _sigint_handler)
        try:
            return super(AnimatedGl, self).start()
        except KeyboardInterrupt:
            return None
        finally:
            for callback in reversed(self.__cleanup_callbacks):
                try:
                    callback(self)
                except Exception:
                    pass
            signal.signal(signal.SIGINT, previous_sigint)
