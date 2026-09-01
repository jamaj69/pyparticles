# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva simone.rva {at} gmail {dot} com
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

import numpy as np


try:
    import pyopencl as cl
    import pyopencl.array as cla
except ImportError as exc:
    cl = None
    cla = None
    _PYOPENCL_IMPORT_ERROR = exc
else:
    _PYOPENCL_IMPORT_ERROR = None


OCLC_X = np.uint8(0b10000000)
OCLC_V = np.uint8(0b01000000)
OCLC_A = np.uint8(0b00100000)
OCLC_M = np.uint8(0b00010000)


class OpenCLcontext(object):
    """Shared OpenCL context and particle buffers.

    Besides owning the OpenCL arrays, the modern implementation tracks which
    side currently contains the newest value for X/V/A/M.  This allows forces
    and integrators sharing one context to avoid uploading a buffer that is
    already current on the device.

    Buffer state is one of:

    ``host``
        The host NumPy array is authoritative and must be uploaded before a
        device consumer uses it.
    ``device``
        The OpenCL array is authoritative and must be downloaded before a host
        consumer uses it.
    ``sync``
        Host and device contain the same value.
    """

    _VALID_NAMES = ("X", "V", "A", "M")

    def __init__(
        self,
        size,
        dim,
        mask=(OCLC_X | OCLC_V | OCLC_A | OCLC_M),
        dtype=np.float32,
    ):
        if cl is None:
            raise RuntimeError("PyOpenCL is not available") from _PYOPENCL_IMPORT_ERROR

        np_dtype = np.dtype(dtype)
        if np_dtype != np.dtype(np.float32):
            raise TypeError("PyParticles OpenCL kernels currently support only float32")

        self.__dtype = np_dtype.type
        self.__size = int(size)
        self.__dim = int(dim)
        self.__opt_arrays = {}

        try:
            self.__cl_context = cl.create_some_context(interactive=False)
        except Exception as exc:
            raise RuntimeError("No usable OpenCL context could be created") from exc

        self.__cl_queue = cl.CommandQueue(
            self.__cl_context,
            properties=cl.command_queue_properties.PROFILING_ENABLE,
        )

        self.__V_cla = self._new_array((self.__size, self.__dim)) if mask & OCLC_V else None
        self.__X_cla = self._new_array((self.__size, self.__dim)) if mask & OCLC_X else None
        self.__A_cla = self._new_array((self.__size, self.__dim)) if mask & OCLC_A else None
        self.__M_cla = self._new_array((self.__size, 1)) if mask & OCLC_M else None

        # At construction the corresponding particle arrays live on the host.
        # Device storage is allocated but not initialized from those arrays yet.
        self.__buffer_state = {
            "X": "host" if self.__X_cla is not None else None,
            "V": "host" if self.__V_cla is not None else None,
            "A": "host" if self.__A_cla is not None else None,
            "M": "host" if self.__M_cla is not None else None,
        }
        self.reset_transfer_stats()

    def _new_array(self, shape, dtype=None):
        if dtype is None:
            dtype = self.__dtype
        return cla.Array(self.__cl_queue, shape, np.dtype(dtype).type)

    def _array_for_name(self, name):
        name = str(name).upper()
        arrays = {
            "X": self.__X_cla,
            "V": self.__V_cla,
            "A": self.__A_cla,
            "M": self.__M_cla,
        }
        if name not in arrays:
            raise KeyError("Unknown OpenCL particle buffer %r" % name)
        array = arrays[name]
        if array is None:
            raise ValueError("OpenCL particle buffer %s was not allocated" % name)
        return name, array

    def _record_transfer(self, direction, name, nbytes):
        self.__transfer_stats[direction + "_calls"] += 1
        self.__transfer_stats[direction + "_bytes"] += int(nbytes)
        self.__transfer_stats["by_buffer"][name][direction + "_calls"] += 1
        self.__transfer_stats["by_buffer"][name][direction + "_bytes"] += int(nbytes)

    def reset_transfer_stats(self):
        self.__transfer_stats = {
            "h2d_calls": 0,
            "d2h_calls": 0,
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "by_buffer": {
                name: {
                    "h2d_calls": 0,
                    "d2h_calls": 0,
                    "h2d_bytes": 0,
                    "d2h_bytes": 0,
                }
                for name in self._VALID_NAMES
            },
        }

    def get_transfer_stats(self):
        return {
            "h2d_calls": self.__transfer_stats["h2d_calls"],
            "d2h_calls": self.__transfer_stats["d2h_calls"],
            "h2d_bytes": self.__transfer_stats["h2d_bytes"],
            "d2h_bytes": self.__transfer_stats["d2h_bytes"],
            "by_buffer": {
                name: dict(values)
                for name, values in self.__transfer_stats["by_buffer"].items()
            },
        }

    transfer_stats = property(get_transfer_stats)

    def get_buffer_state(self, name):
        name, _ = self._array_for_name(name)
        return self.__buffer_state[name]

    def mark_host_modified(self, name):
        name, _ = self._array_for_name(name)
        self.__buffer_state[name] = "host"

    def mark_device_modified(self, name):
        name, _ = self._array_for_name(name)
        self.__buffer_state[name] = "device"

    def mark_synced(self, name):
        name, _ = self._array_for_name(name)
        self.__buffer_state[name] = "sync"

    def sync_to_device(self, name, host_array):
        """Upload *host_array* only when the host contains the newest value."""
        name, device_array = self._array_for_name(name)
        if self.__buffer_state[name] == "host":
            host = np.ascontiguousarray(host_array, dtype=device_array.dtype)
            device_array.set(host, queue=self.__cl_queue)
            self._record_transfer("h2d", name, host.nbytes)
            self.__buffer_state[name] = "sync"
        return device_array

    def sync_to_host(self, name, host_array):
        """Download a device buffer only when the device contains newer data."""
        name, device_array = self._array_for_name(name)
        if self.__buffer_state[name] == "device":
            device_array.get(queue=self.__cl_queue, ary=host_array)
            self._record_transfer("d2h", name, np.asarray(host_array).nbytes)
            self.__buffer_state[name] = "sync"
        return host_array

    def set_from_host(self, name, host_array):
        """Declare the host authoritative and synchronize it to the device."""
        self.mark_host_modified(name)
        return self.sync_to_device(name, host_array)

    def add_array_by_name(self, key, size=None, dim=None, dtype=None):
        if dim is None:
            dim = self.__dim
        if size is None:
            size = self.__size
        if dtype is None:
            dtype = self.dtype

        self.__opt_arrays[key] = self._new_array(
            (int(size), int(dim)),
            dtype=np.dtype(dtype).type,
        )

    def get_by_name(self, key):
        return self.__opt_arrays[key]

    def get_dtype(self):
        return self.__dtype

    dtype = property(get_dtype, doc="return the dtype of the context")

    def get_size(self):
        return self.__size

    size = property(get_size)

    def get_dim(self):
        return self.__dim

    dim = property(get_dim)

    def get_CL_context(self):
        return self.__cl_context

    CL_context = property(get_CL_context, doc="return the opencl context")

    def get_CL_queue(self):
        return self.__cl_queue

    CL_queue = property(get_CL_queue, doc="return the command queue")

    def get_X_cla(self):
        return self.__X_cla

    X_cla = property(get_X_cla, doc="return the positions array")

    def get_A_cla(self):
        return self.__A_cla

    A_cla = property(get_A_cla, doc="return the acceleration array")

    def get_V_cla(self):
        return self.__V_cla

    V_cla = property(get_V_cla, doc="return the velocity array")

    def get_M_cla(self):
        return self.__M_cla

    M_cla = property(get_M_cla, doc="return the masses array")
