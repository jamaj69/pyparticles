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
FreeGLUT/PyOpenGL/NumPy releases.
"""

import ctypes
import signal

import pyparticles.animation.animated_ogl as legacy
import pyparticles.ogl.draw_particles_ogl as legacy_draw
import pyparticles.ogl.draw_particles_ogl_compat as draw_compat
import pyparticles.ogl.draw_vector_field as legacy_vector_field


# The legacy AnimatedGl constructor resolves DrawParticlesGL through the
# ``drp`` module stored in animated_ogl.py.  Replace that constructor before
# any AnimatedGl instance is created, while retaining the original renderer
# module for the remainder of the OpenGL code.
legacy_draw.DrawParticlesGL = draw_compat.DrawParticlesGL
legacy_draw.charged_particles_color = draw_compat.charged_particles_color
legacy_draw.charged_particles_vect_color = draw_compat.charged_particles_vect_color
legacy.drp.DrawParticlesGL = draw_compat.DrawParticlesGL


# OpenGL contexts are commonly destroyed before Python finalizes renderer
# objects.  Avoid issuing glDeleteLists calls from object finalizers after the
# context has disappeared.  The driver releases these objects with the context.
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
    """Legacy renderer with modern FreeGLUT lifecycle compatibility."""

    # The 2012 AnimatedGl getter contains a typo (get_rajectory_step).  Keep
    # the public property working for all demos without editing the renderer.
    def get_trajectory_step(self):
        return legacy.pan.Animation.get_trajectory_step(self)

    def set_trajectory_step(self, value):
        legacy.pan.Animation.set_trajectory_step(self, value)
        self.draw_particles.set_trajectory_step(value)

    trajectory_step = property(get_trajectory_step, set_trajectory_step)

    def build_animation(self):
        # animated_ogl.py uses ctypes.c_char_p() but the original Linux path
        # never imported ctypes.
        legacy.ctypes = ctypes

        # build_animation() assigns ``KeyPressed.animation`` to the callback
        # found in the legacy module, so install our adapter before that step.
        legacy.KeyPressed = _key_pressed

        result = super(AnimatedGl, self).build_animation()
        _configure_freeglut_exit()
        return result

    def start(self):
        previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _sigint_handler)
        try:
            return super(AnimatedGl, self).start()
        except KeyboardInterrupt:
            return None
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
