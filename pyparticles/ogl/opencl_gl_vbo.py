# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# OpenCL/OpenGL zero-host-copy rendering helpers.

"""Share particle positions with OpenGL through a VBO.

The simulation keeps its canonical X array in an ordinary OpenCL buffer.  At
render time this helper acquires an OpenGL VBO through ``cl_khr_gl_sharing``
and performs one device-to-device copy.  No particle positions cross PCIe or
enter a NumPy array in the steady-state rendering path.
"""

import numpy as np

try:
    import pyopencl as cl
except ImportError:
    cl = None

from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_DYNAMIC_DRAW,
    glBindBuffer,
    glBufferData,
    glDeleteBuffers,
    glFinish,
    glGenBuffers,
)


class OpenCLGLPositionBuffer(object):
    """Mirror an OpenCL X buffer into a GL VBO without host transfers."""

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

        vertices = np.ascontiguousarray(pset.X, dtype=np.float32)
        self.__vbo = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self.__vbo)
        glBufferData(GL_ARRAY_BUFFER, self.__nbytes, vertices, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self.__gl_buffer = self.__occ.create_gl_buffer(
            self.__vbo, flags=cl.mem_flags.READ_WRITE
        )

        if draw_particles is not None:
            draw_particles.set_shared_position_vbo(self.__vbo)

    def update_from_device(self):
        """Copy current device X into the shared VBO and return after release."""
        if self.__closed:
            raise RuntimeError("The shared OpenGL position buffer is closed")

        glFinish()
        acquire = self.__occ.acquire_gl_objects([self.__gl_buffer])
        copy = cl.enqueue_copy(
            self.__occ.CL_queue,
            self.__gl_buffer,
            self.__occ.X_cla.data,
            byte_count=self.__nbytes,
            wait_for=[acquire],
        )
        release = self.__occ.release_gl_objects(
            [self.__gl_buffer], wait_for=[copy]
        )
        release.wait()

        self.__copy_calls += 1
        self.__copy_bytes += self.__nbytes
        return release

    def close(self):
        """Release CL and GL views while the owning GL context is still current."""
        if self.__closed:
            return
        self.__closed = True

        # Stop any pending CL work touching the shared object, then make sure
        # the last OpenGL draw has also completed before deleting the VBO.
        try:
            self.__occ.CL_queue.finish()
        except Exception:
            pass
        try:
            glFinish()
        except Exception:
            pass

        if self.__draw_particles is not None:
            try:
                self.__draw_particles.set_shared_position_vbo(None)
            except Exception:
                pass

        try:
            if self.__gl_buffer is not None:
                release = getattr(self.__gl_buffer, "release", None)
                if release is not None:
                    release()
                self.__gl_buffer = None
        except Exception:
            self.__gl_buffer = None

        try:
            if self.__vbo:
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glDeleteBuffers(1, [self.__vbo])
                self.__vbo = 0
        except Exception:
            self.__vbo = 0

    def get_vbo(self):
        return self.__vbo

    vbo = property(get_vbo)

    def get_cl_buffer(self):
        return self.__gl_buffer

    cl_buffer = property(get_cl_buffer)

    def get_copy_stats(self):
        return {
            "calls": self.__copy_calls,
            "bytes": self.__copy_bytes,
        }

    copy_stats = property(get_copy_stats)
