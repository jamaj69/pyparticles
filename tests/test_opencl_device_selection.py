import os
import unittest
from unittest.mock import patch

import pyparticles.pset.opencl_context as opencl_context


class _FakeDevice:
    def __init__(self, name):
        self.name = name


class _FakeContext:
    def __init__(self, devices):
        self.devices = list(devices)


class _FakePlatform:
    def __init__(self, devices):
        self._devices = list(devices)

    def get_devices(self):
        return list(self._devices)


class _FakeCL:
    def __init__(self, selected, platforms):
        self.selected = selected
        self.platforms = list(platforms)
        self.create_some_context_calls = 0

    def create_some_context(self, interactive=False):
        self.create_some_context_calls += 1
        return _FakeContext([self.selected])

    def get_platforms(self):
        return list(self.platforms)


class OpenCLDeviceSelectionTests(unittest.TestCase):
    def _context_shell(self):
        return opencl_context.OpenCLcontext.__new__(opencl_context.OpenCLcontext)

    def test_explicit_pyopencl_ctx_limits_gl_candidates(self):
        nvidia = _FakeDevice("NVIDIA GPU")
        intel = _FakeDevice("Intel CPU")
        fake_cl = _FakeCL(
            selected=intel,
            platforms=[_FakePlatform([nvidia]), _FakePlatform([intel])],
        )

        with patch.object(opencl_context, "cl", fake_cl), patch.dict(
            os.environ, {"PYOPENCL_CTX": "1:0"}, clear=False
        ):
            devices, selector = self._context_shell()._gl_sharing_candidate_devices()

        self.assertEqual(selector, "1:0")
        self.assertEqual(devices, [intel])
        self.assertEqual(fake_cl.create_some_context_calls, 1)

    def test_without_selector_all_devices_remain_candidates(self):
        nvidia = _FakeDevice("NVIDIA GPU")
        intel = _FakeDevice("Intel CPU")
        fake_cl = _FakeCL(
            selected=nvidia,
            platforms=[_FakePlatform([nvidia]), _FakePlatform([intel])],
        )

        env = dict(os.environ)
        env.pop("PYOPENCL_CTX", None)
        with patch.object(opencl_context, "cl", fake_cl), patch.dict(
            os.environ, env, clear=True
        ):
            devices, selector = self._context_shell()._gl_sharing_candidate_devices()

        self.assertIsNone(selector)
        self.assertEqual(devices, [nvidia, intel])
        self.assertEqual(fake_cl.create_some_context_calls, 0)


if __name__ == "__main__":
    unittest.main()
