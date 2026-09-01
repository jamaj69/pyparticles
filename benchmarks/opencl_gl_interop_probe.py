#!/usr/bin/env python3
"""Probe whether the active OpenGL context can share a VBO with PyOpenCL."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pyopencl as cl

from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_DYNAMIC_DRAW,
    glBindBuffer,
    glBufferData,
    glDeleteBuffers,
    glGenBuffers,
)
from OpenGL.GLUT import (
    GLUT_DOUBLE,
    GLUT_RGBA,
    glutCreateWindow,
    glutDestroyWindow,
    glutHideWindow,
    glutInit,
    glutInitDisplayMode,
    glutInitWindowSize,
)


def main():
    print("PyOpenCL GL support:", bool(cl.have_gl()))
    if not cl.have_gl():
        raise SystemExit("PyOpenCL was built without OpenGL interoperability")

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    glutInitWindowSize(32, 32)
    window = glutCreateWindow(b"PyParticles CL/GL interop probe")
    try:
        try:
            glutHideWindow()
        except Exception:
            pass

        payload = np.arange(12, dtype=np.float32)
        vbo = glGenBuffers(1)
        try:
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, payload.nbytes, None, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            sharing = list(cl.get_gl_sharing_context_properties())
            print("GL sharing properties:", sharing)

            last_error = None
            selected = None
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    if "cl_khr_gl_sharing" not in device.extensions.split():
                        continue

                    properties = [
                        (cl.context_properties.PLATFORM, platform),
                        *sharing,
                    ]
                    try:
                        context = cl.Context(devices=[device], properties=properties)
                        queue = cl.CommandQueue(context)
                        gl_buffer = cl.GLBuffer(context, cl.mem_flags.READ_WRITE, int(vbo))
                        source = cl.Buffer(
                            context,
                            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                            hostbuf=payload,
                        )

                        acquire = cl.enqueue_acquire_gl_objects(queue, [gl_buffer])
                        copy = cl.enqueue_copy(
                            queue,
                            gl_buffer,
                            source,
                            byte_count=payload.nbytes,
                            wait_for=[acquire],
                        )
                        release = cl.enqueue_release_gl_objects(
                            queue,
                            [gl_buffer],
                            wait_for=[copy],
                        )
                        release.wait()
                        queue.finish()
                        selected = (platform, device)
                        break
                    except Exception as exc:
                        last_error = exc
                if selected is not None:
                    break

            if selected is None:
                if last_error is not None:
                    raise RuntimeError("No CL/GL sharing context succeeded") from last_error
                raise RuntimeError("No OpenCL device advertises cl_khr_gl_sharing")

            platform, device = selected
            print("Interop platform:", platform.name)
            print("Interop device  :", device.name)
            print("VBO bytes       :", payload.nbytes)
            print("CL/GL interop   : OK")
        finally:
            try:
                glDeleteBuffers(1, [vbo])
            except Exception:
                pass
    finally:
        glutDestroyWindow(window)


if __name__ == "__main__":
    main()
