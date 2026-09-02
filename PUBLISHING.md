# Publishing PyParticles3 to PyPI

PyParticles3 is published through PyPI Trusted Publishing from GitHub Actions.

## Trusted Publishing configuration

The release workflow uses GitHub Actions OIDC instead of storing a long-lived PyPI API token in the repository.

```text
PyPI project name : PyParticles3
GitHub owner      : jamaj69
Repository        : pyparticles
Workflow          : release.yml
Environment       : pypi
```

The GitHub Actions environment used for publication is:

```text
pypi
```

## Published release candidates

The following release candidates have been published and validated:

```text
PyParticles3 0.4.0rc1
PyParticles3 0.4.0rc2
```

`0.4.0rc2` adds explicit diagnostics for PyOpenCL builds without GL interoperability and ensures that an explicit `PYOPENCL_CTX` selection is not silently replaced by another device when CL/GL sharing is attempted.

The `v0.4.0rc2` release was published successfully through Trusted Publishing and validated from a clean PyPI installation.

## Release workflow

`.github/workflows/release.yml` is triggered by version tags and will:

1. build the wheel and source distribution;
2. run `twine check`;
3. upload the built distributions as a workflow artifact;
4. pass them to a separate publish job;
5. request an OIDC publishing credential;
6. upload the release to PyPI.

For a new version, update the package version first, obtain a green CI run, then create a new immutable tag. Never reuse a version already uploaded to PyPI.

For the final `0.4.0`, the intended sequence is:

```bash
# after changing the project version to 0.4.0 and validating CI
git tag -a v0.4.0 -m "PyParticles3 0.4.0"
git push origin v0.4.0
```

If another release candidate is required, use a new version such as `0.4.0rc3`; do not alter or republish `0.4.0rc2`.

## Repository references shown by PyPI

`pyproject.toml` declares:

```text
Homepage       https://github.com/jamaj69/pyparticles
Source         https://github.com/jamaj69/pyparticles
Documentation  https://github.com/jamaj69/pyparticles#readme
Issues         https://github.com/jamaj69/pyparticles/issues
Original       https://github.com/simon-r/PyParticles
```

## Local verification before tagging

Build from a clean source tree:

```bash
python -m pip install -U build twine
rm -rf build dist *.egg-info
python -m build
python -m twine check dist/*
```

Test the wheel in a clean environment and outside the source tree so the checkout cannot shadow the installed package:

```bash
python -m venv /tmp/pyparticles3-release-test
source /tmp/pyparticles3-release-test/bin/activate
cd /tmp
unset PYTHONPATH
python -m pip install '/path/to/dist/PyParticles3-0.4.0rc2-py3-none-any.whl[opencl]'
python -m pip check
pyparticles3 --version
python -m pyparticles --version
pyparticles_app --version
```

For a future version, substitute the actual wheel/version being validated.

## OpenCL compute validation

List all available platform/device pairs:

```bash
python - <<'PY'
import pyopencl as cl
for pi, platform in enumerate(cl.get_platforms()):
    for di, device in enumerate(platform.get_devices()):
        print(f"{pi}:{di}  {platform.name}  {device.name}")
PY
```

Exercise each intended ICD explicitly with `PYOPENCL_CTX`, for example:

```bash
PYOPENCL_CTX=0:0 pyparticles3 --demo fountain
PYOPENCL_CTX=1:0 pyparticles3 --demo fountain
```

The selected compute device must remain selected. A device that cannot share the active OpenGL context may use the host-synchronized rendering fallback, but PyParticles3 must not silently migrate compute to a different device.

## PyOpenCL and OpenGL interoperability

The optional PyParticles3 `opencl` extra guarantees installation of the PyOpenCL Python dependency for OpenCL compute. It does **not** guarantee that PyOpenCL itself was built with OpenGL interoperability.

Check the build:

```bash
python - <<'PY'
import pyopencl as cl
print("PyOpenCL :", cl.VERSION_TEXT)
print("have_gl  :", cl.have_gl())
PY
```

The shared-buffer fountain path requires:

```text
have_gl  : True
```

A standard PyOpenCL wheel may report `False`; that is not an OpenCL-compute failure. In that case PyParticles3 keeps OpenCL compute enabled and uses host synchronization for rendering.

### Build PyOpenCL with GL support

On Debian-family systems, a typical native dependency set is:

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

Then rebuild PyOpenCL from source in the release-test environment:

```bash
python -m pip uninstall -y pyopencl

PYOPENCL_ENABLE_GL=ON \
python -m pip install \
    --no-binary=pyopencl \
    --no-cache-dir \
    -v \
    'pyopencl==2026.1.4'
```

`--no-binary=pyopencl` is required here so pip does not reinstall the precompiled wheel. PyOpenCL's documented source-build switch for GL interoperability is `PYOPENCL_ENABLE_GL=ON`.

Require the result programmatically:

```bash
python - <<'PY'
import pyopencl as cl
print("PyOpenCL :", cl.VERSION_TEXT)
print("have_gl  :", cl.have_gl())
assert cl.have_gl(), "PyOpenCL was built without OpenGL interoperability"
PY
```

`have_gl=True` describes the PyOpenCL build. The selected OpenCL device must additionally support `cl_khr_gl_sharing` and be compatible with the active OpenGL context.

## `0.4.0rc2` validation baseline

The release candidate was validated with both NVIDIA GPU and Intel CPU OpenCL implementations. The NVIDIA CL/GL path was also profiled with 2,000,000 fountain particles on a GeForce GTX 1060 6 GB, producing roughly 278-296 FPS. The Intel CPU path correctly retained the Intel compute device and explicitly fell back to host-synchronized rendering because that device did not advertise `cl_khr_gl_sharing`.

These measurements are regression evidence for the release candidate, not a cross-hardware performance guarantee.
