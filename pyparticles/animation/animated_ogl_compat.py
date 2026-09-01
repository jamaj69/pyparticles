# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Compatibility layer for the legacy PyParticles OpenGL animation.

The renderer itself remains untouched.  This module adapts the pieces whose
runtime behaviour changed between the 2012 PyOpenGL/GLUT stack and current
FreeGLUT/PyOpenGL releases.
"""

import ctypes
import signal

import pyparticles.animation.animated_ogl as legacy


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


class AnimatedGl(legacy.AnimatedGl):
    """Legacy renderer with modern FreeGLUT lifecycle compatibility."""

    def build_animation(self):
        # animated_ogl.py uses ctypes.c_char_p() but the original Linux path
        # never imported ctypes.
        legacy.ctypes = ctypes

        # build_animation() assigns ``KeyPressed.animation`` to the callback
        # found in the legacy module, so install our adapter before that step.
        legacy.KeyPressed = _key_pressed

        return super(AnimatedGl, self).build_animation()

    def start(self):
        previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _sigint_handler)
        try:
            return super(AnimatedGl, self).start()
        except KeyboardInterrupt:
            return None
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
