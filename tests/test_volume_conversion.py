"""Tests for volume conversion utilities."""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pytest

from openlifu.util.volume_conversion import (
    convert_dicom_to_nifti,
    extract_affine_from_dicom,
    is_dicom_file_or_directory,
)


def test_is_dicom_file_or_directory(tmp_path: Path):
    test_dicom_file = Path(__file__).parent / "resources" / "CT_small.dcm"
    assert is_dicom_file_or_directory(test_dicom_file)

    non_dicom_file = tmp_path / "test.nii"
    non_dicom_file.write_bytes(b'not a dicom file')
    assert not is_dicom_file_or_directory(non_dicom_file)

    dicom_dir = tmp_path / "dicom_series"
    dicom_dir.mkdir()
    shutil.copy(test_dicom_file, dicom_dir / "CT_small.dcm")
    assert is_dicom_file_or_directory(dicom_dir)

    non_dicom_dir = tmp_path / "non_dicom"
    non_dicom_dir.mkdir()
    (non_dicom_dir / "file.txt").write_text("not dicom")
    assert not is_dicom_file_or_directory(non_dicom_dir)

    assert not is_dicom_file_or_directory(tmp_path / "nonexistent.dcm")


def test_convert_dicom_to_nifti_single_file(tmp_path: Path):
    test_dicom = Path(__file__).parent / "resources" / "CT_small.dcm"
    output_nifti = tmp_path / "output.nii.gz"

    convert_dicom_to_nifti(test_dicom, output_nifti)

    assert output_nifti.exists()
    img = nib.load(output_nifti)
    assert img.shape[2] == 1  # single slice
    assert not np.allclose(img.affine, np.eye(4))  # affine should not be identity


def test_convert_dicom_to_nifti_directory(tmp_path: Path):
    test_dicom_dir = Path(__file__).parent / "resources" / "dicom_series"
    output_nifti = tmp_path / "output.nii.gz"

    convert_dicom_to_nifti(test_dicom_dir, output_nifti)

    assert output_nifti.exists()
    img = nib.load(output_nifti)
    assert img.shape[2] == 2  # two slices in test series
    assert not np.allclose(img.affine, np.eye(4))  # affine should not be identity


def test_extract_affine_from_dicom():
    # mock DICOM dataset with minimal required tags
    mock_ds = Mock()
    mock_ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    mock_ds.ImagePositionPatient = [10.0, 20.0, 30.0]
    mock_ds.PixelSpacing = [0.5, 0.5]
    mock_ds.SliceThickness = 2.0

    slices = [(1, None, mock_ds)]
    affine = extract_affine_from_dicom(slices)

    assert affine.shape == (4, 4)
    assert np.allclose(affine[3, :], [0, 0, 0, 1])  # homogeneous coordinate
    assert np.allclose(affine[:3, 3], [10.0, 20.0, 30.0])  # position


def test_extract_affine_missing_tags():
    mock_ds = Mock()
    del mock_ds.ImageOrientationPatient  # simulate missing tag

    slices = [(1, None, mock_ds)]
    with pytest.raises(RuntimeError, match="Missing required DICOM tag"):
        extract_affine_from_dicom(slices)
