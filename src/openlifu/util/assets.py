"""Utilities for downloading and installing assets that openlifu needs."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

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
