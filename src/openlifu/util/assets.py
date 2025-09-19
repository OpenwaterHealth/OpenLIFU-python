"""Utilities for downloading and installing assets that openlifu needs."""

from __future__ import annotations

import ast
import importlib
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import requests

from openlifu.util.types import PathLike


def install_asset(destination:PathLike, path_to_asset:PathLike|None, url_to_asset:str|None) -> None:
    """Install a file to a location if it isn't already there.

    Downloads if a `url_to_asset` is provided, and copies if a local `path_to_asset` is provided.
    Does nothing if the `destination` already exists.

    Args:
        destination: The path where the asset should end up. If this already exists then the function will do nothing.
        path_to_asset: Local filepath; if provided then the asset will be copied from here to `destination`.
            Required if url_to_asset is not provided.
        url_to_asset: Web URL to the asset; if provided then the asset will be downloaded and saved to `destination`.
            Required if path_to_asset is not provided.
    """
    destination = Path(destination)

    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    if path_to_asset is not None:
        path_to_asset = Path(path_to_asset)
        shutil.copy2(path_to_asset, destination)
    elif url_to_asset is not None:
        temp_file_path = None
        try:
            response = requests.get(url_to_asset, stream=True, timeout=(10, 300))
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(mode='wb', dir=destination.parent, delete=False) as f:
                temp_file_path = f.name
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            shutil.move(temp_file_path, destination)
        finally:
            if temp_file_path is not None and Path(temp_file_path).exists():
                Path(temp_file_path).unlink()
    else:
        raise ValueError("Either path_to_asset or url_to_asset must be provided.")


def get_modnet_path() -> Path:
    """Get the MODNet checkpoint path.
    It may or may not exist; see `download_and_install_modnet` and `install_modnet_from_file`.
    If `get_modnet_path().exists()` is False, then use one of those two options to install.
    """
    package = "openlifu.nav.modnet_checkpoints"
    filename = "modnet_photographic_portrait_matting.onnx"
    base_dir = Path(importlib.resources.files(package))
    return  base_dir / filename

def download_and_install_modnet() -> Path:
    """Download and install the MODNet checkpoint. Returns path to installed MODNet checkpoint."""
    url = "https://data.kitware.com/api/v1/file/67feb2cb31a330568827ab32/download"
    modnet_path = get_modnet_path()
    install_asset(modnet_path, url_to_asset=url)
    return modnet_path

def install_modnet_from_file(path_to_modnet_file:PathLike) -> Path:
    """Copy MODNet checkpoint to the appropriate place for openlifu to use it.
    Returns path to installed MODNet checkpoint."""
    modnet_path = get_modnet_path()
    install_asset(modnet_path, path_to_asset=path_to_modnet_file)
    return modnet_path

def _import_without_calls(pkg: str, banned_calls:list[str], register=False) -> ModuleType:
    """Import `pkg` but strip any top-level statements that call a banned function.

    It is simplistic: it is looking at the syntax tree and stripping out any node that
    has a banned function call in any of its descendent nodes. There are lots of ways to break
    this if there is enough misdirection in a banned function call. The point of this is just
    to help handle a specific issue we have with kwave's binary download.

    Args:
        pkg: The name of the package to import
        banned_calls: A list of functions to import
        register: Whether to add the module in global import registry.
            Doing so makes any future imports of the module via the usual `import`
            statement end up referring to the version imported here.

    Returns the module.
    """
    spec = importlib.util.find_spec(pkg)
    if not spec or not spec.submodule_search_locations:
        raise ImportError(f"Can't find package {pkg!r}")

    init_path = Path(spec.submodule_search_locations[0]) / "__init__.py"
    src = init_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(init_path))

    # this function tells whether a top level statement tries to call a banned function anywhere inside it
    def stmt_calls_banned(stmt: ast.stmt) -> bool:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name) and f.id in banned_calls:
                    return True
                if isinstance(f, ast.Attribute) and f.attr in banned_calls:
                    return True
        return False

    tree.body = [s for s in tree.body if not stmt_calls_banned(s)] # strip out offending top level statements
    code = compile(tree, str(init_path), "exec")

    module = ModuleType(pkg) # create a blank module object
    module.__file__ = str(init_path)
    module.__package__ = pkg
    g = module.__dict__ # build up the context in which we will execute the module code
    g["__name__"] = pkg
    g["__file__"] = str(init_path)
    exec(code, g, g)

    if register:
        sys.modules[pkg] = module
    return module

def _import_kwave_inertly() -> ModuleType:
    """Import kwave without allowing it to install binaries"""
    return _import_without_calls("kwave", banned_calls=["install_binaries"])

def get_kwave_paths() -> list[tuple[Path, str]]:
    """Get a list of paths and urls to kwave binaries.

    Each item in the list is a pair consisting of the install of a needed binary, followed by a download url for that binary.
    """
    kwave = _import_kwave_inertly()
    paths : list[tuple[str, str]] = []
    for url_list in kwave.URL_DICT[kwave.PLATFORM].values():
        for url in url_list:
            _, filename = url.split("/")[-2:]
            paths.append((Path(kwave.BINARY_PATH) / filename, url))
    return paths
