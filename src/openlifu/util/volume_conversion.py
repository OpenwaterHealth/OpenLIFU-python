"""Utilities for converting between different medical imaging formats."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

from openlifu.util.types import PathLike


def is_dicom_file_or_directory(path: PathLike) -> bool:
    """
    Check if a path is a DICOM file or directory containing DICOM files.

    Args:
        path: Path to check

    Returns:
        True if path is a DICOM file or directory with DICOM files, False otherwise
    """
    path = Path(path)

    if path.is_file():
        # check for 'DICM' magic bytes at offset 128
        try:
            with open(path, 'rb') as f:
                f.seek(128)
                return f.read(4) == b'DICM'
        except OSError:
            return False

    elif path.is_dir():
        for file in path.iterdir():
            if file.is_file():
                try:
                    with open(file, 'rb') as f:
                        f.seek(128)
                        if f.read(4) == b'DICM':
                            return True
                except OSError:
                    continue

    return False


def extract_affine_from_dicom(dicom_slices: list) -> np.ndarray:
    """
    Extract the affine transformation matrix from DICOM header information.

    Args:
        dicom_slices: List of tuples (instance_number, pixel_array, dicom_dataset)

    Returns:
        4x4 affine transformation matrix mapping voxel coordinates to world coordinates

    Raises:
        RuntimeError: If required DICOM tags are missing
    """
    # use the first slice to extract most parameters
    first_ds = dicom_slices[0][2]

    try:
        # ImageOrientationPatient (0020,0037): direction cosines for row and column
        orientation = np.array(first_ds.ImageOrientationPatient, dtype=float)
        row_cosine = orientation[:3]  # direction cosines for rows
        col_cosine = orientation[3:]  # direction cosines for columns

        # ImagePositionPatient (0020,0032): position of the upper-left voxel
        position = np.array(first_ds.ImagePositionPatient, dtype=float)

        # PixelSpacing (0028,0030): spacing between pixels [row_spacing, col_spacing]
        pixel_spacing = np.array(first_ds.PixelSpacing, dtype=float)
        row_spacing = pixel_spacing[0]
        col_spacing = pixel_spacing[1]

    except AttributeError as e:
        raise RuntimeError(
            f"Missing required DICOM tag for affine calculation: {e}"
        ) from e

    # calculate slice direction as cross product of row and column directions
    slice_cosine = np.cross(row_cosine, col_cosine)

    # calculate slice spacing
    if len(dicom_slices) > 1:
        # calculate from the distance between first two slices
        first_pos = np.array(dicom_slices[0][2].ImagePositionPatient, dtype=float)
        second_pos = np.array(dicom_slices[1][2].ImagePositionPatient, dtype=float)
        slice_spacing = np.linalg.norm(second_pos - first_pos)
    else:
        # single slice - try to get from SliceThickness or default to 1.0
        slice_spacing = float(getattr(first_ds, 'SliceThickness', 1.0))

    # construct the affine matrix
    # The affine maps from voxel indices (i, j, k) to physical coordinates (x, y, z)
    affine = np.eye(4)
    affine[:3, 0] = row_cosine * row_spacing
    affine[:3, 1] = col_cosine * col_spacing
    affine[:3, 2] = slice_cosine * slice_spacing
    affine[:3, 3] = position

    return affine


def convert_dicom_to_nifti(input_path: PathLike, output_filepath: PathLike) -> None:
    """
    Convert DICOM file(s) to NIfTI format using pydicom and nibabel.

    The affine transformation matrix is extracted from DICOM headers to properly
    map voxel coordinates to scanner coordinates in the NIfTI output.

    Args:
        input_path: Path to either a DICOM file or directory containing DICOM files
        output_filepath: Path where the output NIfTI file should be saved

    Raises:
        RuntimeError: If the conversion fails
    """
    input_path = Path(input_path)
    output_filepath = Path(output_filepath)

    try:
        if input_path.is_file():
            dicom_files = [input_path]
        else:
            # dicom files may not have .dcm extension
            dicom_files = [f for f in input_path.iterdir() if f.is_file()]

        if not dicom_files:
            raise RuntimeError("No DICOM files found")

        slices = []
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(dcm_file)
                # store instance number, pixel array, and dataset for affine extraction
                slices.append((ds.get('InstanceNumber', 0), ds.pixel_array, ds))
            except Exception:
                # skip files that aren't valid dicom
                continue

        if not slices:
            raise RuntimeError("No valid DICOM files found")

        # sort by instance number - this is the slice order in the series
        # so we reconstruct the 3D volume in the right order
        slices.sort(key=lambda x: x[0])

        # stack into 3D volume (handles both single and multiple slices)
        volume = np.stack([s[1] for s in slices], axis=-1)

        # extract affine from DICOM headers
        affine = extract_affine_from_dicom(slices)

        nifti_img = nib.Nifti1Image(volume, affine)
        nib.save(nifti_img, str(output_filepath))

    except Exception as e:
        raise RuntimeError(f"DICOM to NIfTI conversion failed: {e}") from e
