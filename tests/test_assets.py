from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from openlifu.util.assets import get_kwave_paths, install_asset


def test_destination_already_exists(tmp_path, mocker):
    """Test that nothing happens if the destination file already exists."""
    destination = tmp_path / "asset.dat"
    destination.write_text("original content")

    spy_copy = mocker.spy(shutil, "copy2")
    spy_get = mocker.spy(requests, "get")

    install_asset(destination, path_to_asset="dummy/path", url_to_asset="http://dummy.url")

    assert spy_copy.call_count == 0
    assert spy_get.call_count == 0
    assert destination.read_text() == "original content"

def test_local_copy_succeeds(tmp_path):
    """Test that a local file is copied correctly."""
    source_file = tmp_path / "source.txt"
    source_file.write_text("local asset data")
    destination = tmp_path / "installed" / "asset.txt"

    install_asset(destination, path_to_asset=source_file, url_to_asset=None)

    assert destination.exists()
    assert destination.read_text() == "local asset data"

def test_download_succeeds(tmp_path, mocker):
    """Test a successful asset download and installation."""
    url = "http://example.com/asset.zip"
    destination = tmp_path / "asset.zip"
    fake_content = b"\x01\x02\x03\x04\x05"

    mock_response = MagicMock()
    mock_response.iter_content.return_value = [fake_content]
    # Make raise_for_status do nothing
    mock_response.raise_for_status.return_value = None

    mock_get = mocker.patch("requests.get", return_value=mock_response)

    install_asset(destination, path_to_asset=None, url_to_asset=url)

    mock_get.assert_called_once_with(url, stream=True, timeout=(10, 300))
    assert destination.exists()
    assert destination.read_bytes() == fake_content

def test_download_fails_on_http_error(tmp_path, mocker):
    """Test that an HTTP error is raised and no files are left behind."""
    url = "http://example.com/notfound.zip"
    destination = tmp_path / "assets" / "notfound.zip"

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")

    mocker.patch("requests.get", return_value=mock_response)

    with pytest.raises(requests.exceptions.HTTPError):
        install_asset(destination, path_to_asset=None, url_to_asset=url)

    # Assert that the parent directory might exist, but is empty
    assert not destination.exists()
    assert not any(destination.parent.iterdir())

def test_download_cleans_up_on_interruption(tmp_path, mocker):
    """Test that a mid-download interruption cleans up the temporary file."""
    url = "http://example.com/largefile.zip"
    destination = tmp_path / "assets" / "largefile.zip"
    expected_error_message = "Network connection broken"

    mock_response = MagicMock()
    # Simulate an error after the first chunk is read
    mock_response.iter_content.side_effect = OSError(expected_error_message)
    mock_response.raise_for_status.return_value = None

    mocker.patch("requests.get", return_value=mock_response)

    with pytest.raises(IOError, match=expected_error_message):
        install_asset(destination, path_to_asset=None, url_to_asset=url)

    assert not destination.exists()
    assert not any(destination.parent.iterdir())

def test_raises_error_if_no_source_provided(tmp_path):
    """Test that a ValueError is raised if no source is given."""
    destination = tmp_path / "asset.dat"
    with pytest.raises(ValueError, match="Either path_to_asset or url_to_asset must be provided."):
        install_asset(destination, path_to_asset=None, url_to_asset=None)

def test_get_kwave_paths():
    """Check that get_kwave_paths returns a nonempty list of (Path, str) pairs.
    This may break if kwave changes how it represents download urls and binary paths."""
    paths = get_kwave_paths()
    assert isinstance(paths, list)
    assert len(paths) > 0
    for p, url in paths:
        assert isinstance(p, Path)
        assert isinstance(url, str)
        assert len(url) > 0
