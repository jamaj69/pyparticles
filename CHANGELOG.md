# Changelog

All notable changes to the modernized distribution are documented here.

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

- validated OpenCL gravity kernels on NVIDIA GTX 1060 and Intel Xeon E5-2697 v2 OpenCL implementations;
- validated CPU/OpenCL numerical agreement over complete Euler integration runs;
- validated persistent X/V device residency and transfer accounting;
- validated clean PyPI installation and all published CLI entry points;
- validated the high-performance fountain CL/GL path with a GL-enabled PyOpenCL build.

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
