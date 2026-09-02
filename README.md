# PyParticles3

**PyParticles3** is an independent modernization and continuation of Simone Riva's original **PyParticles** project.

The goal is to preserve the original project's unusually clear, educational architecture while updating it for modern Python, NumPy, SciPy, PyOpenGL and PyOpenCL environments. PyParticles3 is also a practical study project for GPU particle simulation, OpenCL acceleration and OpenCL/OpenGL interoperability.

> PyParticles3 is not presented as an official release endorsed by Simone Riva. Original copyright notices and the GPL-3.0-or-later license are preserved.

## Highlights

- simple `ParticlesSet -> Force -> Solver -> Animation/Renderer` architecture;
- Euler, Leapfrog, Runge-Kutta, Midpoint and Stormer-Verlet style integrators;
- gravity, springs, constant force, drag, damping, Lennard-Jones, electrostatic and electromagnetic models;
- constrained particles and constrained force interactions;
- OpenGL interactive visualization;
- optional PyOpenCL acceleration;
- persistent OpenCL buffers to avoid unnecessary PCIe transfers;
- fused OpenCL integration paths;
- OpenCL/OpenGL shared VBO rendering;
- double-buffered CL/GL synchronization with GL fences;
- asynchronous OpenGL GPU timer queries;
- an optimized `fountain` example used as a GPU performance case study.

## Installation

The current release candidate is `PyParticles3 0.4.0rc2`.

```bash
python -m pip install PyParticles3
```

For OpenCL compute support:

```bash
python -m pip install 'PyParticles3[opencl]'
```

The OpenCL runtime/ICD and an OpenGL/FreeGLUT implementation are system-level dependencies and are not installed by pip.

### OpenCL/OpenGL interoperability requires a GL-enabled PyOpenCL build

`PyParticles3[opencl]` installs PyOpenCL and is sufficient for OpenCL **compute**. It does **not** guarantee that the installed PyOpenCL binary was compiled with OpenGL interoperability enabled.

Check the installed build with:

```bash
python - <<'PY'
import pyopencl as cl
print("PyOpenCL:", cl.VERSION_TEXT)
print("have_gl :", cl.have_gl())
PY
```

For the fast OpenCL/OpenGL shared-buffer path used by the `fountain` demo, the required result is:

```text
have_gl : True
```

If a PyPI wheel reports `have_gl : False`, OpenCL compute still works, but PyParticles3 cannot create `GLBuffer` objects and high-particle-count rendering must fall back to host synchronization. That fallback can be dramatically slower.

PyOpenCL's source-build documentation requires `PYOPENCL_ENABLE_GL=ON` to enable GL interoperability. A typical pip rebuild is:

```bash
python -m pip uninstall -y pyopencl

PYOPENCL_ENABLE_GL=ON \
python -m pip install \
    --no-binary=pyopencl \
    --no-cache-dir \
    -v \
    'pyopencl>=2026.1'
```

Then verify again:

```bash
python - <<'PY'
import pyopencl as cl
assert cl.have_gl(), "PyOpenCL was built without OpenGL interoperability"
print("PyOpenCL GL interoperability: enabled")
PY
```

Building PyOpenCL from source also requires a C++17 compiler, Python/build dependencies, OpenCL headers and loader libraries, and OpenGL development headers appropriate to the operating system.

## Selecting an OpenCL device

PyParticles3 follows PyOpenCL's `PYOPENCL_CTX` selection for compute contexts. For example, on a system where platform 0 is an NVIDIA GPU and platform 1 is an Intel CPU:

```bash
PYOPENCL_CTX=0:0 pyparticles3 --demo fountain
PYOPENCL_CTX=1:0 pyparticles3 --demo fountain
```

As of `0.4.0rc2`, an explicit `PYOPENCL_CTX` selection is also honored when PyParticles3 attempts to create a CL/GL sharing context. PyParticles3 will not silently move the simulation to another OpenCL device just because that other device supports `cl_khr_gl_sharing`.

This matters on mixed systems. An Intel Xeon CPU OpenCL device can execute the simulation kernels but may not advertise or support `cl_khr_gl_sharing` with the active NVIDIA OpenGL context. In that case PyParticles3 keeps the selected Intel compute device and falls back to host-synchronized rendering instead of silently switching compute to the NVIDIA GPU.

## Current import namespace

The initial PyParticles3 packaging release intentionally keeps the historical Python import namespace:

```python
import pyparticles
```

This avoids mixing a package-wide namespace migration with the packaging release. A future release may introduce a dedicated `pyparticles3` namespace after a controlled compatibility migration.

## Command line

The modern package exposes:

```bash
pyparticles3 --help
pyparticles3 --version
```

The historical command is also kept as a compatibility entry point:

```bash
pyparticles_app --help
```

Examples:

```bash
pyparticles3 --demo springs
pyparticles3 --demo solar_system
pyparticles3 --demo bubble
pyparticles3 --demo gas_lj
pyparticles3 --demo elmag_field
pyparticles3 --demo galaxy
pyparticles3 --demo fountain
```

## Architecture

```text
ParticlesSet
    |
    +--> Force / MultipleForce
    |        |
    |        v
    +----> ODE Solver
             |
             v
        Animation
             |
             v
          Renderer
```

The accelerated paths preserve these conceptual roles rather than replacing the whole program with an opaque GPU pipeline.

## OpenCL/OpenGL fountain path

The modern `fountain` demo can keep simulation state resident on the GPU and render from shared OpenGL buffers. The optimized fused path writes both the canonical OpenCL position buffer and the shared render VBO from the integration kernel, eliminating a separate device-to-device position copy.

Profiling can be enabled with:

```bash
PYPARTICLES_PROFILE_CLGL=1 \
PYPARTICLES_PROFILE_FRAMES=1000 \
PYPARTICLES_PROFILE_WARMUP=200 \
pyparticles3 --demo fountain
```

The experimental fused render mirror is currently selected with:

```bash
PYPARTICLES_CLGL_FUSED_MIRROR=1 \
pyparticles3 --demo fountain
```

## Development

```bash
git clone https://github.com/jamaj69/pyparticles.git
cd pyparticles
git switch package/pyparticles3

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'

python -m compileall -q pyparticles tests
python -W default -m unittest discover -v -s tests
```

Build the PyPI artifacts with:

```bash
python -m build
python -m twine check dist/*
```

## Project links

- Source: https://github.com/jamaj69/pyparticles
- Issues: https://github.com/jamaj69/pyparticles/issues
- Original project: https://github.com/simon-r/PyParticles

When releases are published through PyPI Trusted Publishing from this GitHub repository, PyPI can verify the GitHub project links carried in the distribution metadata.

## Origin and attribution

PyParticles was created by **Simone Riva** in 2012. PyParticles3 is an independent modernization built from that GPL-licensed codebase. The modernization focuses on Python 3 compatibility, modern scientific Python libraries, testing, GPU acceleration, CL/GL interoperability and documentation while preserving the original educational structure.

## License

GPL-3.0-or-later. See `LICENSE-gpl-3.0.txt`.
