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

The stable release line is **PyParticles3 0.4.0**. The Python import namespace remains `pyparticles`.

PyParticles3 requires Python 3.11 or newer.

### 1. Standard installation

Install PyParticles3 from PyPI:

```bash
python -m pip install --upgrade pip
python -m pip install 'PyParticles3==0.4.0'
```

Verify the installation:

```bash
pyparticles3 --version
python -m pyparticles --version
```

Both commands should report:

```text
0.4.0
```

The historical console command remains available for compatibility:

```bash
pyparticles_app --version
```

### 2. Installation with OpenCL compute support

Install the optional OpenCL dependency set with:

```bash
python -m pip install 'PyParticles3[opencl]==0.4.0'
```

This installs PyOpenCL, but it does **not** install a system OpenCL driver/ICD. A working NVIDIA, Intel, AMD, PoCL or other OpenCL runtime must already be installed on the operating system.

List the OpenCL platforms and devices visible to PyOpenCL:

```bash
python - <<'PY'
import pyopencl as cl

for pi, platform in enumerate(cl.get_platforms()):
    print(f"Platform {pi}: {platform.name}")
    for di, device in enumerate(platform.get_devices()):
        print(f"  Device {di}: {device.name}")
PY
```

PyParticles3 follows PyOpenCL's `PYOPENCL_CTX` selector. For example:

```bash
PYOPENCL_CTX=0:0 pyparticles3 --demo fountain
PYOPENCL_CTX=1:0 pyparticles3 --demo fountain
```

An explicitly selected compute device is never silently replaced by a different OpenCL device merely to obtain OpenCL/OpenGL sharing.

### 3. Check whether PyOpenCL has OpenGL interoperability

The PyPI PyOpenCL wheel can provide fully working OpenCL compute while still being built **without** OpenGL interoperability. Check the installed build explicitly:

```bash
python - <<'PY'
import pyopencl as cl

print("PyOpenCL :", cl.VERSION_TEXT)
print("Module   :", cl.__file__)
print("have_gl  :", cl.have_gl())
PY
```

For ordinary OpenCL compute, either value of `have_gl()` is acceptable.

For the high-performance OpenCL/OpenGL shared-buffer path used by the `fountain` demo, PyOpenCL itself must report:

```text
have_gl  : True
```

If it reports:

```text
have_gl  : False
```

OpenCL compute still works, but PyParticles3 cannot create PyOpenCL `GLBuffer` objects. Rendering then uses host synchronization and can be dramatically slower for large particle counts.

### 4. Build PyOpenCL from source with `have_gl=True`

PyOpenCL's source-build documentation requires the build option `PYOPENCL_ENABLE_GL=ON` to enable OpenGL interoperability.

#### Debian 12 / Debian-family build prerequisites

A typical Debian installation can provide the native build dependencies with:

```bash
sudo apt update
sudo apt install \
    build-essential \
    python3-dev \
    cmake \
    ninja-build \
    pkg-config \
    ocl-icd-opencl-dev \
    libgl-dev \
    freeglut3-dev
```

`ocl-icd-opencl-dev` supplies the OpenCL development headers and loader needed to compile against the system OpenCL installation. Your actual OpenCL implementation/ICD, such as the NVIDIA or Intel runtime, remains a separate system component.

If Python comes from pyenv or another custom Python installation, make sure that installation includes its matching Python headers; the system `python3-dev` package applies to Debian's system Python.

#### Rebuild PyOpenCL

Inside the same virtual environment in which PyParticles3 is installed:

```bash
python -m pip uninstall -y pyopencl

PYOPENCL_ENABLE_GL=ON \
python -m pip install \
    --no-binary=pyopencl \
    --no-cache-dir \
    -v \
    'pyopencl==2026.1.4'
```

`--no-binary=pyopencl` is important: it forces a source build instead of reinstalling the precompiled wheel.

The `0.4.0` release line was qualified with PyOpenCL `2026.1.4`. Newer compatible PyOpenCL releases can also be built from source, but should be tested before being used as a release-validation baseline.

#### Verify the resulting build

Do not assume that a successful compilation enabled GL support. Require it explicitly:

```bash
python - <<'PY'
import pyopencl as cl

print("PyOpenCL :", cl.VERSION_TEXT)
print("Module   :", cl.__file__)
print("have_gl  :", cl.have_gl())

assert cl.have_gl(), "PyOpenCL was built without OpenGL interoperability"
PY
```

The final line must be:

```text
have_gl  : True
```

#### Verify the PyParticles3 CL/GL path

Select an OpenCL GPU that can share the active OpenGL context and run:

```bash
PYOPENCL_CTX=0:0 pyparticles3 --demo fountain
```

A successful shared-buffer path reports messages similar to:

```text
OpenCL/OpenGL interop enabled: positions render without host copies
CL/GL sync: double-buffered VBOs with per-buffer GL fences
CL/GL position path: X -> VBO device copy (stable)
Interop device: NVIDIA GeForce ...
```

`pyopencl.have_gl() == True` means that the **PyOpenCL build** contains GL interoperability support. It does not guarantee that every OpenCL device can share the current OpenGL context. The selected device must also support `cl_khr_gl_sharing` and be compatible with the active GL context.

For example, on a mixed NVIDIA-GPU/Intel-CPU system, an Intel CPU OpenCL device may run all simulation kernels correctly but not advertise `cl_khr_gl_sharing`. PyParticles3 then keeps the Intel device selected and explicitly falls back to host-synchronized rendering instead of moving the computation to NVIDIA.

### 5. System OpenGL requirements

Interactive rendering requires a working OpenGL implementation and FreeGLUT. These are operating-system dependencies and are not installed by pip.

On Debian-family systems, the development packages used above include the common OpenGL/FreeGLUT headers. The graphics driver must still provide a working OpenGL runtime.

### 6. Install from the Git repository

For development or testing the current repository state:

```bash
git clone https://github.com/jamaj69/pyparticles.git
cd pyparticles

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For development with OpenCL support:

```bash
python -m pip install -e '.[dev,opencl]'
```

If CL/GL interoperability is required, rebuild PyOpenCL with `PYOPENCL_ENABLE_GL=ON` after installing the editable package, using the procedure above.

## Current import namespace

The PyParticles3 0.4.x release line intentionally keeps the historical Python import namespace:

```python
import pyparticles
```

This avoids mixing a package-wide namespace migration with the first modernized stable release. A future release may introduce a dedicated `pyparticles3` namespace after a controlled compatibility migration.

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

The modern `fountain` demo can keep simulation state resident on the GPU and render from shared OpenGL buffers.

The stable default shared path copies positions from the canonical OpenCL position buffer into a shared OpenGL VBO entirely on the device. The code also contains an optional experimental fused render-mirror path that can write the shared render VBO directly from the integration kernel.

Profiling can be enabled with:

```bash
PYPARTICLES_PROFILE_CLGL=1 \
PYPARTICLES_PROFILE_FRAMES=1000 \
PYPARTICLES_PROFILE_WARMUP=200 \
pyparticles3 --demo fountain
```

The experimental fused render mirror is selected with:

```bash
PYPARTICLES_CLGL_FUSED_MIRROR=1 \
pyparticles3 --demo fountain
```

The final `0.4.0` release is based on the validated `0.4.0rc2` code path. The release-candidate baseline on a GeForce GTX 1060 6 GB with 2,000,000 fountain particles produced roughly 278-296 FPS, with the fused physics kernel around 0.760 ms and the device-side X-to-VBO copy around 0.324 ms. Treat these numbers as a hardware-specific regression baseline, not as a general performance guarantee.

## Development

```bash
git clone https://github.com/jamaj69/pyparticles.git
cd pyparticles

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'

python -m compileall -q pyparticles tests
python -W default -m unittest discover -v -s tests
```

Build the PyPI artifacts with:

```bash
python -m pip install -U build twine
rm -rf build dist *.egg-info
python -m build
python -m twine check dist/*
```

## Project links

- Source: https://github.com/jamaj69/pyparticles
- Issues: https://github.com/jamaj69/pyparticles/issues
- Original project: https://github.com/simon-r/PyParticles
- PyOpenCL installation/build documentation: https://documen.tician.de/pyopencl/misc.html

When releases are published through PyPI Trusted Publishing from this GitHub repository, PyPI can verify the GitHub project links carried in the distribution metadata.

## Origin and attribution

PyParticles was created by **Simone Riva** in 2012. PyParticles3 is an independent modernization built from that GPL-licensed codebase. The modernization focuses on Python 3 compatibility, modern scientific Python libraries, testing, GPU acceleration, CL/GL interoperability and documentation while preserving the original educational structure.

## License

GPL-3.0-or-later. See `LICENSE-gpl-3.0.txt`.
