from __future__ import annotations

import importlib.metadata

import openlifu as m


def test_version():
    assert importlib.metadata.version("openlifu") == m.__version__
