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

The first PyPI release is planned as `PyParticles3 0.4.0rc1`.

Once published:

```bash
python -m pip install PyParticles3
```

For OpenCL support:

```bash
python -m pip install 'PyParticles3[opencl]'
```

The OpenCL runtime/ICD and an OpenGL/FreeGLUT implementation are system-level dependencies and are not installed by pip.

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
