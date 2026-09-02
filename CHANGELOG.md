# Changelog

All notable changes to the modernized distribution are documented here.

## 0.4.0 - 2026-09-01

First stable PyParticles3 release after the `0.4.0rc1` and `0.4.0rc2` validation cycle.

### Release status

- promote the validated `0.4.0rc2` code path to the stable `0.4.0` version;
- keep the historical Python import namespace `pyparticles` while publishing the distribution as `PyParticles3`;
- keep the `pyparticles3`, `pyparticles_app`, and `python -m pyparticles` entry points;
- retain Python 3.11, 3.12, and 3.13 CI coverage;
- retain PyPI Trusted Publishing through GitHub Actions OIDC.

### Installation and OpenCL documentation

- make the stable PyPI installation commands explicit;
- distinguish PyOpenCL OpenCL-compute support from OpenCL/OpenGL interoperability;
- document `pyopencl.have_gl()` verification;
- document Debian-family native build prerequisites;
- document the validated PyOpenCL 2026.1.4 source-build procedure using `PYOPENCL_ENABLE_GL=ON` and `--no-binary=pyopencl`;
- document that an explicit `PYOPENCL_CTX` selection remains authoritative even when the selected device cannot use `cl_khr_gl_sharing`.

### Validation inherited from rc2

The final release is based on the rc2 code path that passed:

- clean PyPI installation and CLI smoke tests;
- OpenCL gravity and integration validation on NVIDIA GPU and Intel CPU implementations;
- persistent device-residency and transfer-accounting tests;
- NVIDIA OpenCL/OpenGL shared-buffer rendering;
- Intel explicit-device fallback without silent migration to NVIDIA;
- a 2,000,000-particle fountain baseline of roughly 278-296 FPS on a GeForce GTX 1060 6 GB, with fused physics around 0.760 ms and the device-side X-to-VBO copy around 0.324 ms.

No physics algorithm or OpenCL kernel behavior is intentionally changed between the validated rc2 code path and the final `0.4.0` release preparation.

## 0.4.0rc2 - release candidate

Follow-up release candidate based on installation and multi-ICD validation of `0.4.0rc1`.

### OpenCL/OpenGL interoperability

- document that `PyParticles3[opencl]` guarantees OpenCL compute support but does not guarantee a PyOpenCL build with OpenGL interoperability;
- add an explicit runtime warning when `pyopencl.have_gl()` is `False`;
- document rebuilding PyOpenCL from source with `PYOPENCL_ENABLE_GL=ON` and verifying `pyopencl.have_gl() is True`;
- preserve an explicit `PYOPENCL_CTX` device selection when creating a CL/GL sharing context;
- do not silently migrate an explicitly selected Intel CPU OpenCL workload to an NVIDIA GPU merely to obtain `cl_khr_gl_sharing`;
- improve CL/GL failure messages so host-synchronized fallback identifies why sharing is unavailable.

### Validation

- published `PyParticles3 0.4.0rc2` to PyPI using GitHub Actions Trusted Publishing/OIDC;
- validated a clean PyPI installation and all published CLI entry points;
- validated that the standard PyOpenCL wheel can report `have_gl=False` and that PyParticles3 emits the intended diagnostic while retaining OpenCL compute support;
- rebuilt PyOpenCL 2026.1.4 from source with `PYOPENCL_ENABLE_GL=ON` and validated `pyopencl.have_gl() is True`;
- validated OpenCL gravity kernels on NVIDIA GTX 1060 and Intel Xeon E5-2697 v2 OpenCL implementations;
- validated CPU/OpenCL numerical agreement over complete Euler integration runs;
- validated persistent X/V device residency and transfer accounting;
- validated the high-performance fountain CL/GL path on NVIDIA;
- validated that `PYOPENCL_CTX=1:0` keeps Intel selected and explicitly falls back to host synchronization when the Intel device does not advertise `cl_khr_gl_sharing`;
- validated a 2,000,000-particle fountain baseline on a GeForce GTX 1060 6 GB at roughly 278-296 FPS, with the fused physics kernel around 0.760 ms and X-to-VBO device copy around 0.324 ms.

## 0.4.0rc1 - release candidate

First release candidate under the **PyParticles3** distribution name.

### Packaging

- replace legacy `distutils` metadata with `pyproject.toml` and `setuptools.build_meta`;
- add wheel and source-distribution builds;
- add `pyparticles3` console entry point while preserving `pyparticles_app`;
- add `python -m pyparticles` entry point;
- declare PyPI project URLs pointing to the modern GitHub repository;
- add PyPI Trusted Publishing workflow using GitHub Actions OIDC;
- add CI build/install smoke tests;
- preserve GPL-3.0-or-later licensing and original Simone Riva attribution.

### Python and scientific stack modernization

- Python 3 compatibility work;
- NumPy 2 compatibility fixes;
- modern SciPy sparse API usage;
- modern PyOpenGL/FreeGLUT compatibility layer;
- modern PyOpenCL API compatibility.

### OpenCL

- persistent device buffers and host/device residency tracking;
- accelerated gravity, drag, Euler and selected demo paths;
- compute-only modes that avoid unnecessary host synchronization;
- fused fountain force/integration/boundary kernel.

### OpenCL/OpenGL interoperability

- OpenGL VBOs shared with OpenCL through `cl_khr_gl_sharing`;
- double-buffered shared position VBOs;
- per-buffer OpenGL fences;
- optional fused simulation-to-render VBO mirror;
- deterministic cleanup of shared resources before GL context shutdown.

### Profiling and rendering

- OpenCL event profiling;
- asynchronous `GL_TIME_ELAPSED` particle draw profiling;
- controlled OpenGL benchmark modes;
- `no_msaa` optimized default for the fountain demo;
- CL/GL frame profiler for reproducible performance analysis.

### Tests

- compatibility and numerical regression tests;
- OpenCL compute tests when a usable device is available;
- fused fountain render-mirror correctness test;
- cleanup callback lifetime regression test.
