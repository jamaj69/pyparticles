# Changelog

All notable changes to the modernized distribution are documented here.

## 0.4.0rc1 - release candidate

First release candidate planned under the **PyParticles3** distribution name.

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
