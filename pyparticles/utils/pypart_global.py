# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Project-wide version and optional dependency probes."""

import zlib


v_major = 0
v_minor = 4
v_revision = 0
v_prerelease = "rc1"


def py_particle_version(r="s"):
    """Return the PyParticles3 compatibility version.

    The string form follows the PyPI distribution version.  The historical
    tuple form remains a three-integer tuple for compatibility with callers
    that expect ``(major, minor, revision)``.
    """
    if r == "s":
        return "%d.%d.%d%s" % (v_major, v_minor, v_revision, v_prerelease)
    return (v_major, v_minor, v_revision)


def test_pyopencl():
    """Return True only when PyOpenCL has at least one usable device."""
    try:
        import pyopencl as cl
    except ImportError:
        return False

    try:
        return any(platform.get_devices() for platform in cl.get_platforms())
    except Exception:
        # Broken/missing ICDs may allow importing pyopencl while still making
        # every OpenCL operation unusable.
        return False


def about():
    mail = zlib.decompress(
        b"x\x9c+\xce\xcc\xcd\xcfK\xd5+*KtH\xcfM\xcc\xcc\xd1K\xce\xcf\x05\x00R\x9c\x07\xba"
    ).decode("utf-8")

    message = """

    PyParticles3 is an independent modernization of the original PyParticles
    particle simulation toolbox created by Simone Riva.

    The project preserves the original educational architecture while adding
    modern Python compatibility, tests, OpenCL acceleration, OpenCL/OpenGL
    interoperability, GPU profiling, and updated documentation.

    Modern source: https://github.com/jamaj69/pyparticles
    Original source: https://github.com/simon-r/PyParticles

    Original copyright (C) 2012 Simone Riva, email: %s

    --------------------------------------------------------------------

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.
    """

    print(message % mail)
