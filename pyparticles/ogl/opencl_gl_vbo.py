# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# OpenCL/OpenGL zero-host-copy rendering helpers.

"""Share particle positions with OpenGL through double-buffered VBOs.

The simulation keeps its canonical X array in an ordinary OpenCL buffer.  At
render time this helper copies X into one of two OpenGL VBOs exposed through
``cl_khr_gl_sharing``.  Each VBO receives a GL fence after drawing; OpenCL only
waits for the particular VBO it is about to reuse instead of globally stalling
all OpenGL work with ``glFinish()`` every frame.
"""

import time

import numpy as np

try:
    import pyopencl as cl
except ImportError:
    cl = None

from OpenGL.GL import (
    GL_ALREADY_SIGNALED,
    GL_ARRAY_BUFFER,
    GL_CONDITION_SATISFIED,
    GL_DYNAMIC_DRAW,
    GL_SYNC_FLUSH_COMMANDS_BIT,
    GL_SYNC_GPU_COMMANDS_COMPLETE,
    GL_TIMEOUT_EXPIRED,
    GL_WAIT_FAILED,
    glBindBuffer,
    glBufferData,
    glClientWaitSync,
    glDeleteBuffers,
    glDeleteSync,
    glFenceSync,
    glFinish,
    glGenBuffers,
)


def _event_profile_seconds(event):
    """Return valid timing phases for a completed OpenCL event.

    NVIDIA's GL acquire/release events may expose ``start``/``end`` while
    leaving ``queued`` or ``submit`` equal to zero.  Treat unavailable phases
    as NaN instead of subtracting zero from an absolute device timestamp.
    """
    nan = float("nan")
    result = {
        "queued_to_submit_s": nan,
        "submit_to_start_s": nan,
        "queued_to_start_s": nan,
        "execution_s": nan,
        "queued_to_end_s": nan,
    }
    if event is None:
        return result

    try:
        queued = int(event.profile.queued)
        submit = int(event.profile.submit)
        start = int(event.profile.start)
        end = int(event.profile.end)
    except Exception:
        return result

    scale = 1.0e-9
    if queued > 0 and submit >= queued:
        result["queued_to_submit_s"] = (submit - queued) * scale
    if submit > 0 and start >= submit:
        result["submit_to_start_s"] = (start - submit) * scale
    if queued > 0 and start >= queued:
        result["queued_to_start_s"] = (start - queued) * scale
    if start > 0 and end >= start:
        result["execution_s"] = (end - start) * scale
    if queued > 0 and end >= queued:
        result["queued_to_end_s"] = (end - queued) * scale
    return result


class OpenCLGLPositionBuffer(object):
    """Mirror an OpenCL X buffer into alternating GL VBOs without host copies."""

    _BUFFER_COUNT = 2
    _FENCE_TIMEOUT_NS = 100000000  # 100 ms; fallback to glFinish after this.

    def __init__(self, ocl_context, pset, draw_particles=None):
        if cl is None:
            raise RuntimeError("PyOpenCL is required for CL/GL interoperability")
        if not getattr(ocl_context, "gl_sharing", False):
            raise RuntimeError("OpenCL context was not created with GL sharing")

        self.__occ = ocl_context
        self.__pset = pset
        self.__draw_particles = draw_particles
        self.__nbytes = int(pset.size * pset.dim * np.dtype(np.float32).itemsize)
        self.__copy_calls = 0
        self.__copy_bytes = 0
        self.__closed = False
        self.__next_index = 0
        self.__active_index = 0
        self.__vbos = []
        self.__gl_buffers = []
        self.__fences = [None] * self._BUFFER_COUNT
        self.__needs_gl_completion = [False] * self._BUFFER_COUNT
        self.__vbo_to_index = {}
        self.__last_profile = {
            "gl_finish_wall_s": 0.0,
            "gl_fence_wait_wall_s": 0.0,
            "gl_finish_fallback_wall_s": 0.0,
            "fence_immediate": 1.0,
            "acquire_gpu_s": 0.0,
            "acquire_queue_s": float("nan"),
            "acquire_total_s": float("nan"),
            "copy_gpu_s": 0.0,
            "copy_queue_s": float("nan"),
            "copy_total_s": float("nan"),
            "release_gpu_s": 0.0,
            "release_queue_s": float("nan"),
            "release_total_s": float("nan"),
            "release_wait_wall_s": 0.0,
            "bridge_wall_s": 0.0,
            "vbo_index": 0.0,
        }

        vertices = np.ascontiguousarray(pset.X, dtype=np.float32)
        try:
            for index in range(self._BUFFER_COUNT):
                vbo = int(glGenBuffers(1))
                self.__vbos.append(vbo)
                self.__vbo_to_index[vbo] = index
                glBindBuffer(GL_ARRAY_BUFFER, vbo)
                glBufferData(
                    GL_ARRAY_BUFFER,
                    self.__nbytes,
                    vertices,
                    GL_DYNAMIC_DRAW,
                )
                self.__gl_buffers.append(
                    self.__occ.create_gl_buffer(
                        vbo, flags=cl.mem_flags.READ_WRITE
                    )
                )
        finally:
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        if len(self.__vbos) != self._BUFFER_COUNT:
            self.close()
            raise RuntimeError("Could not allocate both CL/GL position VBOs")

        if draw_particles is not None:
            draw_particles.set_shared_position_vbo(self.__vbos[0])
            draw_particles.set_shared_position_draw_complete_callback(
                self.mark_draw_complete
            )

    def mark_draw_complete(self, vbo):
        """Insert a fence immediately after OpenGL submits a draw using *vbo*."""
        if self.__closed:
            return

        index = self.__vbo_to_index.get(int(vbo))
        if index is None:
            return

        old_fence = self.__fences[index]
        if old_fence is not None:
            try:
                glDeleteSync(old_fence)
            except Exception:
                pass
            self.__fences[index] = None

        self.__needs_gl_completion[index] = True
        try:
            fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
            if fence:
                self.__fences[index] = fence
        except Exception:
            self.__fences[index] = None

    def _wait_for_gl_vbo(self, index):
        """Wait only for the GL draw that last consumed one VBO."""
        if not self.__needs_gl_completion[index]:
            return 0.0, 1.0, 0.0

        start = time.perf_counter()
        fallback_seconds = 0.0
        immediate = 0.0
        fence = self.__fences[index]

        try:
            if fence is None:
                fallback_start = time.perf_counter()
                glFinish()
                fallback_seconds = time.perf_counter() - fallback_start
            else:
                result = glClientWaitSync(fence, 0, 0)
                if result in (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED):
                    immediate = 1.0
                else:
                    result = glClientWaitSync(
                        fence,
                        GL_SYNC_FLUSH_COMMANDS_BIT,
                        self._FENCE_TIMEOUT_NS,
                    )
                    if result not in (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED):
                        fallback_start = time.perf_counter()
                        glFinish()
                        fallback_seconds = time.perf_counter() - fallback_start
        except Exception:
            fallback_start = time.perf_counter()
            glFinish()
            fallback_seconds = time.perf_counter() - fallback_start
        finally:
            if fence is not None:
                try:
                    glDeleteSync(fence)
                except Exception:
                    pass
            self.__fences[index] = None
            self.__needs_gl_completion[index] = False

        return time.perf_counter() - start, immediate, fallback_seconds

    def update_from_device(self):
        """Copy current device X into the next free VBO and select it for draw."""
        if self.__closed:
            raise RuntimeError("The shared OpenGL position buffer is closed")

        bridge_start = time.perf_counter()
        index = self.__next_index
        self.__next_index = (self.__next_index + 1) % self._BUFFER_COUNT

        fence_wait, fence_immediate, finish_fallback = self._wait_for_gl_vbo(index)

        gl_buffer = self.__gl_buffers[index]
        acquire = self.__occ.acquire_gl_objects([gl_buffer])
        copy = cl.enqueue_copy(
            self.__occ.CL_queue,
            gl_buffer,
            self.__occ.X_cla.data,
            byte_count=self.__nbytes,
            wait_for=[acquire],
        )
        release = self.__occ.release_gl_objects(
            [gl_buffer], wait_for=[copy]
        )

        # Without cl_khr_gl_event, OpenGL must not consume the VBO until the CL
        # release is complete.  Keep this wait so rendering remains correct;
        # profiling below exposes whether the delay is queue latency or actual
        # ownership/copy execution.
        wait_start = time.perf_counter()
        release.wait()
        wait_end = time.perf_counter()

        acquire_profile = _event_profile_seconds(acquire)
        copy_profile = _event_profile_seconds(copy)
        release_profile = _event_profile_seconds(release)

        self.__active_index = index
        if self.__draw_particles is not None:
            self.__draw_particles.set_shared_position_vbo(self.__vbos[index])

        bridge_end = time.perf_counter()
        self.__last_profile = {
            "gl_finish_wall_s": 0.0,
            "gl_fence_wait_wall_s": fence_wait,
            "gl_finish_fallback_wall_s": finish_fallback,
            "fence_immediate": fence_immediate,
            "acquire_gpu_s": acquire_profile["execution_s"],
            "acquire_queue_s": acquire_profile["queued_to_start_s"],
            "acquire_total_s": acquire_profile["queued_to_end_s"],
            "copy_gpu_s": copy_profile["execution_s"],
            "copy_queue_s": copy_profile["queued_to_start_s"],
            "copy_total_s": copy_profile["queued_to_end_s"],
            "release_gpu_s": release_profile["execution_s"],
            "release_queue_s": release_profile["queued_to_start_s"],
            "release_total_s": release_profile["queued_to_end_s"],
            "release_wait_wall_s": wait_end - wait_start,
            "bridge_wall_s": bridge_end - bridge_start,
            "vbo_index": float(index),
        }

        self.__copy_calls += 1
        self.__copy_bytes += self.__nbytes
        return release

    def close(self):
        """Release CL and GL views while the owning GL context is still current."""
        if self.__closed:
            return
        self.__closed = True

        occ = self.__occ

        # Teardown is not a hot path.  Fully drain both APIs before destroying
        # fences and shared objects so the NVIDIA/GLX lifetime ordering remains
        # deterministic.
        try:
            if occ is not None:
                occ.CL_queue.finish()
        except Exception:
            pass
        try:
            glFinish()
        except Exception:
            pass

        if self.__draw_particles is not None:
            try:
                self.__draw_particles.set_shared_position_draw_complete_callback(None)
                self.__draw_particles.set_gpu_timing_enabled(False)
                self.__draw_particles.set_shared_position_vbo(None)
            except Exception:
                pass

        for index, fence in enumerate(self.__fences):
            if fence is not None:
                try:
                    glDeleteSync(fence)
                except Exception:
                    pass
                self.__fences[index] = None

        for gl_buffer in self.__gl_buffers:
            try:
                release = getattr(gl_buffer, "release", None)
                if release is not None:
                    release()
            except Exception:
                pass
        self.__gl_buffers = []

        for vbo in self.__vbos:
            try:
                if vbo:
                    glDeleteBuffers(1, [vbo])
            except Exception:
                pass
        self.__vbos = []
        self.__vbo_to_index = {}

        self.__occ = None
        self.__pset = None
        self.__draw_particles = None

    def get_vbo(self):
        if not self.__vbos:
            return 0
        return self.__vbos[self.__active_index]

    vbo = property(get_vbo)

    def get_cl_buffer(self):
        if not self.__gl_buffers:
            return None
        return self.__gl_buffers[self.__active_index]

    cl_buffer = property(get_cl_buffer)

    def get_copy_stats(self):
        return {
            "calls": self.__copy_calls,
            "bytes": self.__copy_bytes,
        }

    copy_stats = property(get_copy_stats)

    def get_last_profile(self):
        """Return timings for the most recent CL/GL bridge update."""
        return dict(self.__last_profile)

    last_profile = property(get_last_profile)
