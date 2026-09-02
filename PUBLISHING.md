# Publishing PyParticles3 to PyPI

This repository is prepared to publish the `PyParticles3` distribution through PyPI Trusted Publishing.

## Why Trusted Publishing

The release workflow uses GitHub Actions OIDC instead of storing a long-lived PyPI API token in the repository. PyPI issues a short-lived credential to the trusted workflow at publication time.

## PyPI publisher configuration

The PyPI project is published from this GitHub repository with these values:

```text
PyPI project name : PyParticles3
GitHub owner      : jamaj69
Repository        : pyparticles
Workflow          : release.yml
Environment       : pypi
```

## GitHub environment

The release workflow uses the GitHub Actions environment:

```text
pypi
```

For a public release project, required reviewers on this environment are recommended so a tag cannot publish without an explicit approval step.

## Release candidates

The first published candidate was:

```text
PyParticles3 0.4.0rc1
```

The current candidate is:

```text
PyParticles3 0.4.0rc2
```

`0.4.0rc2` adds explicit diagnostics for PyOpenCL builds without GL interoperability and ensures that an explicit `PYOPENCL_CTX` selection is not silently replaced by another device when CL/GL sharing is attempted.

After CI is green, publish the candidate with:

```bash
git tag -a v0.4.0rc2 -m "PyParticles3 0.4.0rc2"
git push origin v0.4.0rc2
```

`.github/workflows/release.yml` will:

1. build the wheel and source distribution;
2. run `twine check`;
3. pass the artifacts to a separate publish job;
4. request an OIDC publishing credential;
5. upload the release to PyPI.

## Repository references shown by PyPI

`pyproject.toml` declares:

```text
Homepage       https://github.com/jamaj69/pyparticles
Source         https://github.com/jamaj69/pyparticles
Documentation  https://github.com/jamaj69/pyparticles#readme
Issues         https://github.com/jamaj69/pyparticles/issues
Original       https://github.com/simon-r/PyParticles
```

Because releases are uploaded through GitHub Trusted Publishing from `jamaj69/pyparticles`, PyPI can verify links belonging to that GitHub repository.

## Release discipline

Do not reuse a version already uploaded to PyPI. If a release artifact must change, increment the version first.

Recommended sequence:

```text
0.4.0rc1
0.4.0rc2
0.4.0rc3   (only if another release candidate is needed)
0.4.0
```

## Local verification before tagging

```bash
python -m pip install -U build twine
rm -rf build dist *.egg-info
python -m build
python -m twine check dist/*
```

Then test the wheel in a clean environment, outside the source tree:

```bash
python -m venv /tmp/pyparticles3-release-test
source /tmp/pyparticles3-release-test/bin/activate
cd /tmp
unset PYTHONPATH
python -m pip install '/path/to/dist/PyParticles3-0.4.0rc2-py3-none-any.whl[opencl]'
python -m pip check
pyparticles3 --version
```

## OpenCL compute validation

List available platforms/devices and test each desired ICD explicitly. For example:

```bash
python - <<'PY'
import pyopencl as cl
for pi, platform in enumerate(cl.get_platforms()):
    for di, device in enumerate(platform.get_devices()):
        print(f"{pi}:{di}  {platform.name}  {device.name}")
PY
```

Then select devices with `PYOPENCL_CTX`, for example:

```bash
PYOPENCL_CTX=0:0 pyparticles3 --demo fountain
PYOPENCL_CTX=1:0 pyparticles3 --demo fountain
```

The selected device must remain the selected compute device. `0.4.0rc2` must not silently migrate an explicit Intel CPU selection to an NVIDIA GPU merely to obtain CL/GL sharing.

## CL/GL release validation

The optional `opencl` extra guarantees PyOpenCL compute support only. A PyOpenCL wheel may report:

```text
have_gl : False
```

For CL/GL validation, verify:

```bash
python - <<'PY'
import pyopencl as cl
print("PyOpenCL:", cl.VERSION_TEXT)
print("have_gl :", cl.have_gl())
PY
```

The shared-buffer fountain path requires:

```text
have_gl : True
```

If necessary, rebuild PyOpenCL from source with GL support:

```bash
python -m pip uninstall -y pyopencl

PYOPENCL_ENABLE_GL=ON \
python -m pip install \
    --no-binary=pyopencl \
    --no-cache-dir \
    -v \
    'pyopencl>=2026.1'
```

Then require the result programmatically:

```bash
python - <<'PY'
import pyopencl as cl
assert cl.have_gl(), "PyOpenCL was built without OpenGL interoperability"
print("PyOpenCL GL interoperability: enabled")
PY
```

A selected OpenCL device that cannot share the active OpenGL context is allowed to fall back to host-synchronized rendering, but the fallback must be explicit and the compute device must not silently change.
