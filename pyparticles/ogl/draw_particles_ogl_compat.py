# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# Modern compatibility helpers for the legacy OpenGL particle renderer.

"""Modern PyOpenGL/NumPy compatibility for :mod:`draw_particles_ogl`.

The original renderer assumes Python-2-era scalar coercions and frequently
passes float64 NumPy storage to OpenGL pointers declared as GL_FLOAT.  This
module preserves its public interface while normalizing scalar values and
client arrays for current PyOpenGL and NumPy releases.
"""

import numpy as np

from OpenGL.GL import (
    GL_COLOR_ARRAY,
    GL_FLOAT,
    GL_LINES,
    GL_POINTS,
    GL_UNSIGNED_INT,
    GL_VERTEX_ARRAY,
    glBegin,
    glCallList,
    glColor4f,
    glColorPointer,
    glDisableClientState,
    glDrawArrays,
    glDrawElements,
    glEnableClientState,
    glEnd,
    glPointSize,
    glPopMatrix,
    glPushMatrix,
    glScalef,
    glTranslatef,
    glVertex3f,
    glVertexPointer,
)

import pyparticles.ogl.draw_particles_ogl as legacy


def _scalar(value):
    """Return a Python float from a scalar or one-element NumPy value."""
    return float(np.asarray(value).reshape(-1)[0])


def charged_particles_color(pset, i):
    a = 0.4
    charge = _scalar(pset.Q[i])
    if charge > 0.0:
        return (1.0, a, a, 1.0)
    if charge < 0.0:
        return (a, a, 1.0, 1.0)
    return (a, a, a, 1.0)


def charged_particles_vect_color(RGBA, pset):
    a = 0.4
    charges = np.asarray(pset.Q).reshape(-1)
    RGBA[charges > 0.0, :] = (1.0, a, a, 1.0)
    RGBA[charges < 0.0, :] = (a, a, 1.0, 1.0)
    RGBA[charges == 0.0, :] = (a, a, a, 1.0)


class DrawParticlesGL(legacy.DrawParticlesGL):
    """Legacy particle renderer with modern scalar and buffer handling."""

    def __del__(self):
        # OpenGL contexts are frequently already gone during interpreter
        # teardown.  The driver releases display-list resources with context
        # destruction, so avoid unsafe GL calls from __del__.
        pass

    def _get_color_fun(self):
        return self._DrawParticlesGL__color_fun

    def _set_color_fun(self, fun):
        self._DrawParticlesGL__color_fun = fun

    color_fun = property(_get_color_fun, _set_color_fun)

    def _get_vect_color_fun(self):
        return self._DrawParticlesGL__vect_color_fun

    def _set_vect_color_fun(self, fun):
        self._DrawParticlesGL__vect_color_fun = fun
        self._DrawParticlesGL__vect_color_fun_fl = False

    vect_color_fun = property(_get_vect_color_fun, _set_vect_color_fun)

    def draw_particle(self, pset, i):
        mass = _scalar(pset.M[i])
        glPointSize(float(0.01 + mass / pset.mass_unit))
        glColor4f(*self._DrawParticlesGL__color_fun(pset, i))

        glBegin(GL_POINTS)
        glVertex3f(
            float(pset.X[i, 0] / pset.unit),
            float(pset.X[i, 1] / pset.unit),
            float(pset.X[i, 2] / pset.unit),
        )
        glEnd()

    def draw_particle_sphere(self, pset, i):
        mass = _scalar(pset.M[i])
        radius = 0.5 * (0.05 + 0.1 / (1.0 + np.exp(-mass / pset.mass_unit)))

        glColor4f(*self._DrawParticlesGL__color_fun(pset, i))
        glPushMatrix()
        glTranslatef(
            float(pset.X[i, 0] / pset.unit),
            float(pset.X[i, 1] / pset.unit),
            float(pset.X[i, 2] / pset.unit),
        )
        glScalef(float(radius), float(radius), float(radius))
        glCallList(self._DrawParticlesGL__sph_dl)
        glPopMatrix()

    def draw_particle_teapot(self, pset, i):
        mass = _scalar(pset.M[i])
        radius = 0.5 * (0.05 + 0.1 / (1.0 + np.exp(-mass / pset.mass_unit)))

        glColor4f(*self._DrawParticlesGL__color_fun(pset, i))
        glPushMatrix()
        glTranslatef(
            float(pset.X[i, 0] / pset.unit),
            float(pset.X[i, 1] / pset.unit),
            float(pset.X[i, 2] / pset.unit),
        )
        glScalef(float(radius), float(radius), float(radius))
        glCallList(self._DrawParticlesGL__tea_dl)
        glPopMatrix()

    def _draw_vectorized(self):
        vertices = np.ascontiguousarray(
            np.asarray(self.pset.X) / self.pset.unit,
            dtype=np.float32,
        )

        vect_color_fun = self._DrawParticlesGL__vect_color_fun
        if vect_color_fun is not None:
            colors = np.empty((self.pset.size, 4), dtype=np.float32)
            vect_color_fun(colors, self.pset)
            colors = np.ascontiguousarray(colors, dtype=np.float32)
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(4, GL_FLOAT, 0, colors)
        else:
            colors = None

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        glDrawArrays(GL_POINTS, 0, self.pset.size)
        glDisableClientState(GL_VERTEX_ARRAY)

        if colors is not None:
            glDisableClientState(GL_COLOR_ARRAY)

    def draw_trajectory(self):
        if self.pset.log_size < self.trajectory_step + 1:
            return

        indices = self._DrawParticlesGL__log_indices
        if indices is None or len(indices) != max(0, 2 * self.pset.log_max_size - 2):
            indices = self.pset.get_log_indices_segments(True)
            self._DrawParticlesGL__log_indices = indices
            self._DrawParticlesGL__log_array = np.zeros(
                (self.pset.log_max_size, self.pset.dim),
                dtype=np.float32,
            )

        log_array = self._DrawParticlesGL__log_array
        for i in range(self.pset.size):
            glColor4f(*self._DrawParticlesGL__color_fun(self.pset, i))
            glEnableClientState(GL_VERTEX_ARRAY)

            _, count = self.pset.read_log_array(i, (log_array,))
            if count > 0:
                vertices = np.ascontiguousarray(
                    log_array / self.pset.unit,
                    dtype=np.float32,
                )
                glVertexPointer(3, GL_FLOAT, 0, vertices)
                glDrawElements(
                    GL_LINES,
                    count,
                    GL_UNSIGNED_INT,
                    np.ascontiguousarray(indices[:count], dtype=np.uint32),
                )

            glDisableClientState(GL_VERTEX_ARRAY)


# Keep color helpers obtained from the historical module working as well.
legacy.charged_particles_color = charged_particles_color
legacy.charged_particles_vect_color = charged_particles_vect_color
