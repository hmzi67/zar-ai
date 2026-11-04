"""Every module must import on core + dev: optional heavy deps load inside functions."""

import importlib
import pkgutil

import pytest

import zarai

MODULES = [m.name for m in pkgutil.walk_packages(zarai.__path__, "zarai.")]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
