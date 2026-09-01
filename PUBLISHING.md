# Publishing PyParticles3 to PyPI

This repository is prepared to publish the `PyParticles3` distribution through PyPI Trusted Publishing.

## Why Trusted Publishing

The release workflow uses GitHub Actions OIDC instead of storing a long-lived PyPI API token in the repository. PyPI issues a short-lived credential to the trusted workflow at publication time.

## PyPI pending publisher configuration

Before the first release, sign in to PyPI and add a **pending GitHub publisher** with these exact values:

```text
PyPI project name : PyParticles3
GitHub owner      : jamaj69
Repository        : pyparticles
Workflow          : release.yml
Environment       : pypi
```

The pending publisher does not reserve the project name until the first successful publication.

## GitHub environment

Create a GitHub Actions environment named:

```text
pypi
```

For a public release project, enabling required reviewers on this environment is recommended so a tag cannot publish without an explicit approval step.

## First release

The package metadata is currently:

```text
PyParticles3 0.4.0rc1
```

After CI is green and the pending publisher is configured, create and push the matching tag:

```bash
git tag -a v0.4.0rc1 -m "PyParticles3 0.4.0rc1"
git push origin v0.4.0rc1
```

`.github/workflows/release.yml` will:

1. build the wheel and source distribution;
2. run `twine check`;
3. pass the artifacts to a separate publish job;
4. request an OIDC publishing credential;
5. upload the release to PyPI.

On first successful use of a pending publisher, PyPI creates the project and converts the publisher to a normal trusted publisher.

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
0.4.0rc2   (only if another release candidate is needed)
0.4.0
```

## Local verification before tagging

```bash
python -m pip install -U build twine
rm -rf build dist *.egg-info
python -m build
python -m twine check dist/*
```

Then test the wheel in a clean environment:

```bash
python -m venv /tmp/pyparticles3-release-test
source /tmp/pyparticles3-release-test/bin/activate
python -m pip install dist/*.whl
pyparticles3 --version
```
