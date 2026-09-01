"""Compatibility shim for legacy tooling.

Modern builds are configured entirely in pyproject.toml.  Keeping this tiny
setup.py allows older editable-install tooling to delegate to setuptools
without duplicating project metadata.
"""

from setuptools import setup


setup()
