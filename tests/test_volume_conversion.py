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
    # Ensure 3D shape even for single slice
    assert img.shape[2] == 1
    # Affine should definitely not be identity for a valid medical image
    assert not np.allclose(img.affine, np.eye(4))


def test_convert_dicom_to_nifti_directory(tmp_path: Path):
    test_dicom_dir = Path(__file__).parent / "resources" / "dicom_series"
    output_nifti = tmp_path / "output.nii.gz"

    convert_dicom_to_nifti(test_dicom_dir, output_nifti)

    assert output_nifti.exists()
    img = nib.load(output_nifti)
    assert img.shape[2] == 2
    assert not np.allclose(img.affine, np.eye(4))


def test_extract_affine_standard_orientation():
    """
    Test affine extraction for a standard axial slice.
    Verifies proper mapping of Row/Col spacing and LPS->RAS conversion.
    """
    mock_ds = Mock()
    # Standard Axial: Row is X, Col is Y
    mock_ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # Position in LPS (Left, Posterior, Superior)
    mock_ds.ImagePositionPatient = [10.0, 20.0, 30.0]
    # PixelSpacing: [RowSpacing (dy), ColSpacing (dx)]
    mock_ds.PixelSpacing = [0.5, 0.8]
    mock_ds.SliceThickness = 2.0

    slices = [(1, None, mock_ds)]
    affine = extract_affine_from_dicom(slices)

    # Expected RAS Matrix:
    # 1. Spacing: X=0.8, Y=0.5, Z=2.0
    # 2. Orientation: Standard Axial means X aligns with Right, Y with Anterior.
    #    However, DICOM is LPS.
    #    LPS X+ (Left) converts to RAS X- (Left).
    #    LPS Y+ (Posterior) converts to RAS Y- (Posterior).
    # 3. Origin: [10, 20, 30] LPS -> [-10, -20, 30] RAS

    expected_affine = np.array([
        [-0.8,  0.0,  0.0, -10.0],
        [ 0.0, -0.5,  0.0, -20.0],
        [ 0.0,  0.0,  2.0,  30.0],
        [ 0.0,  0.0,  0.0,   1.0]
    ])

    np.testing.assert_allclose(affine, expected_affine)


def test_extract_affine_multi_slice_spacing():
    """Test that Z-spacing is calculated from slice positions, not SliceThickness."""
    ds1 = Mock()
    ds1.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds1.PixelSpacing = [1.0, 1.0]
    ds1.ImagePositionPatient = [0.0, 0.0, 10.0]  # z = 10

    ds2 = Mock()
    ds2.ImagePositionPatient = [0.0, 0.0, 12.5]  # z = 12.5

    # Create slice list with dummy pixel arrays
    slices = [
        (1, None, ds1),
        (2, None, ds2)
    ]

    affine = extract_affine_from_dicom(slices)

    # Z spacing should be 12.5 - 10.0 = 2.5
    assert np.isclose(affine[2, 2], 2.5)
    # Origin should be from the first slice, converted to RAS (Z is unchanged)
    assert np.isclose(affine[2, 3], 10.0)


def test_extract_affine_missing_tags():
    mock_ds = Mock()
    del mock_ds.ImageOrientationPatient  # simulate missing tag

    slices = [(1, None, mock_ds)]
    with pytest.raises(RuntimeError, match="Missing required DICOM tag"):
        extract_affine_from_dicom(slices)
