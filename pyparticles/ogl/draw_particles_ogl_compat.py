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

import ctypes
import time

import numpy as np

from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_ARRAY,
    GL_FLOAT,
    GL_LINES,
    GL_POINTS,
    GL_QUERY_RESULT,
    GL_QUERY_RESULT_AVAILABLE,
    GL_TIME_ELAPSED,
    GL_UNSIGNED_INT,
    GL_VERTEX_ARRAY,
    glBegin,
    glBeginQuery,
    glBindBuffer,
    glCallList,
    glColor4f,
    glColorPointer,
    glDeleteQueries,
    glDisableClientState,
    glDrawArrays,
    glDrawElements,
    glEnableClientState,
    glEnd,
    glEndQuery,
    glGenQueries,
    glPointSize,
    glPopMatrix,
    glPushMatrix,
    glScalef,
    glTranslatef,
    glVertex3f,
    glVertexPointer,
)
from OpenGL.raw.GL.VERSION.GL_1_5 import (
    glGetQueryObjectiv as _raw_glGetQueryObjectiv,
)
from OpenGL.raw.GL.VERSION.GL_3_3 import (
    glGetQueryObjectui64v as _raw_glGetQueryObjectui64v,
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

    _GPU_QUERY_LIMIT = 64

    def __init__(self, *args, **kwargs):
        super(DrawParticlesGL, self).__init__(*args, **kwargs)
        self.__shared_position_vbo = None
        self.__shared_position_draw_complete_callback = None
        self.__last_draw_submit_seconds = 0.0

        # GL timer queries are opt-in and are polled asynchronously.  Never ask
        # for GL_QUERY_RESULT until GL_QUERY_RESULT_AVAILABLE says the result is
        # ready, otherwise the profiler itself would insert a GPU/CPU stall.
        self.__gpu_timing_enabled = False
        self.__gpu_timing_available = True
        self.__gpu_timing_error = None
        self.__gpu_queries_pending = []
        self.__gpu_draw_seconds_ready = []
        self.__gpu_query_skipped = 0

    def __del__(self):
        # OpenGL contexts are frequently already gone during interpreter
        # teardown.  The driver releases display-list/query resources with
        # context destruction, so avoid unsafe GL calls from __del__.
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

    def set_shared_position_vbo(self, vbo):
        """Use *vbo* as the vectorized particle position source.

        ``None`` restores the normal host NumPy client-array path.  The VBO is
        expected to contain tightly-packed float32 XYZ triples in particle
        order and remains owned by the CL/GL bridge that created it.
        """
        self.__shared_position_vbo = None if vbo is None else int(vbo)

    def get_shared_position_vbo(self):
        return self.__shared_position_vbo

    shared_position_vbo = property(
        get_shared_position_vbo, set_shared_position_vbo
    )

    def set_shared_position_draw_complete_callback(self, callback):
        """Set a callback invoked just after a shared-VBO draw is submitted."""
        self.__shared_position_draw_complete_callback = callback

    def get_shared_position_draw_complete_callback(self):
        return self.__shared_position_draw_complete_callback

    shared_position_draw_complete_callback = property(
        get_shared_position_draw_complete_callback,
        set_shared_position_draw_complete_callback,
    )

    def get_last_draw_submit_seconds(self):
        """CPU time spent submitting the most recent glDrawArrays call."""
        return self.__last_draw_submit_seconds

    last_draw_submit_seconds = property(get_last_draw_submit_seconds)

    def set_gpu_timing_enabled(self, enabled):
        """Enable non-blocking GL_TIME_ELAPSED measurements for glDrawArrays."""
        enabled = bool(enabled)
        if enabled:
            self.__gpu_timing_enabled = True
            return

        self.__gpu_timing_enabled = False
        self.cleanup_gpu_timing()

    def get_gpu_timing_enabled(self):
        return self.__gpu_timing_enabled

    gpu_timing_enabled = property(
        get_gpu_timing_enabled, set_gpu_timing_enabled
    )

    def _disable_gpu_timing(self, exc):
        self.__gpu_timing_available = False
        self.__gpu_timing_error = "%s: %s" % (
            exc.__class__.__name__, exc
        )

    def _poll_gpu_timing(self):
        """Collect completed timer queries without waiting for the GPU.

        Use raw OpenGL entry points with explicit ctypes output storage.  The
        high-level PyOpenGL wrappers for glGetQueryObject* are not consistently
        auto-wrapped on all installations and may require the output pointer.
        """
        if not self.__gpu_queries_pending:
            return

        while self.__gpu_queries_pending:
            query = self.__gpu_queries_pending[0]
            try:
                available_value = ctypes.c_int(0)
                _raw_glGetQueryObjectiv(
                    query,
                    GL_QUERY_RESULT_AVAILABLE,
                    ctypes.byref(available_value),
                )
                available = bool(available_value.value)
            except Exception as exc:
                self._disable_gpu_timing(exc)
                return

            if not available:
                break

            self.__gpu_queries_pending.pop(0)
            try:
                elapsed_value = ctypes.c_uint64(0)
                _raw_glGetQueryObjectui64v(
                    query,
                    GL_QUERY_RESULT,
                    ctypes.byref(elapsed_value),
                )
                self.__gpu_draw_seconds_ready.append(
                    int(elapsed_value.value) * 1.0e-9
                )
            except Exception as exc:
                self._disable_gpu_timing(exc)
            finally:
                try:
                    glDeleteQueries(1, [query])
                except Exception:
                    pass

    def poll_gpu_timing(self):
        self._poll_gpu_timing()

    def drain_gpu_draw_times(self):
        """Return completed draw timings accumulated since the previous drain."""
        self._poll_gpu_timing()
        values = list(self.__gpu_draw_seconds_ready)
        self.__gpu_draw_seconds_ready[:] = []
        return values

    def get_gpu_timing_stats(self):
        return {
            "available": bool(self.__gpu_timing_available),
            "pending": len(self.__gpu_queries_pending),
            "ready": len(self.__gpu_draw_seconds_ready),
            "skipped": int(self.__gpu_query_skipped),
            "error": self.__gpu_timing_error,
        }

    gpu_timing_stats = property(get_gpu_timing_stats)

    def cleanup_gpu_timing(self):
        """Delete pending query objects while the GL context is still current."""
        for query in self.__gpu_queries_pending:
            try:
                glDeleteQueries(1, [query])
            except Exception:
                pass
        self.__gpu_queries_pending[:] = []
        self.__gpu_draw_seconds_ready[:] = []

    def _draw_arrays_profiled(self):
        """Submit the point draw, optionally wrapped in an asynchronous timer."""
        self._poll_gpu_timing()
        query = None

        if (
            self.__gpu_timing_enabled
            and self.__gpu_timing_available
            and len(self.__gpu_queries_pending) < self._GPU_QUERY_LIMIT
        ):
            try:
                query = int(glGenQueries(1))
                glBeginQuery(GL_TIME_ELAPSED, query)
            except Exception as exc:
                self._disable_gpu_timing(exc)
                query = None
        elif self.__gpu_timing_enabled:
            self.__gpu_query_skipped += 1

        draw_start = time.perf_counter()
        try:
            glDrawArrays(GL_POINTS, 0, self.pset.size)
        finally:
            self.__last_draw_submit_seconds = time.perf_counter() - draw_start
            if query is not None:
                try:
                    glEndQuery(GL_TIME_ELAPSED)
                    self.__gpu_queries_pending.append(query)
                except Exception as exc:
                    self._disable_gpu_timing(exc)
                    try:
                        glDeleteQueries(1, [query])
                    except Exception:
                        pass

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
        vect_color_fun = self._DrawParticlesGL__vect_color_fun
        if vect_color_fun is not None:
            colors = np.empty((self.pset.size, 4), dtype=np.float32)
            vect_color_fun(colors, self.pset)
            colors = np.ascontiguousarray(colors, dtype=np.float32)
            # Client color memory must be captured while no array buffer is
            # bound; otherwise the pointer is interpreted as a VBO offset.
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(4, GL_FLOAT, 0, colors)
        else:
            colors = None

        glEnableClientState(GL_VERTEX_ARRAY)
        if self.__shared_position_vbo is None:
            vertices = np.ascontiguousarray(
                np.asarray(self.pset.X) / self.pset.unit,
                dtype=np.float32,
            )
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glVertexPointer(3, GL_FLOAT, 0, vertices)
            self._draw_arrays_profiled()
        else:
            # Positions are already in GPU memory.  Apply the unit conversion
            # as a model-view scale instead of materializing a host array.
            vbo = self.__shared_position_vbo
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
            glPushMatrix()
            unit_scale = 1.0 / float(self.pset.unit)
            glScalef(unit_scale, unit_scale, unit_scale)
            self._draw_arrays_profiled()

            # The fence belongs immediately after the command that consumes
            # this VBO.  The CL/GL bridge uses it before reacquiring this same
            # buffer on a later frame, avoiding a global glFinish().
            callback = self.__shared_position_draw_complete_callback
            if callback is not None:
                callback(vbo)

            glPopMatrix()
            glBindBuffer(GL_ARRAY_BUFFER, 0)

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
                glBindBuffer(GL_ARRAY_BUFFER, 0)
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
