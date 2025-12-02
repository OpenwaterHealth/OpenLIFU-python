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


def extract_affine_from_dicom(
    dicom_slices: list[tuple[int, np.ndarray, pydicom.Dataset]]
) -> np.ndarray:
    """
    Extract the affine transformation matrix from DICOM header information.
    Converts from DICOM LPS (Left-Posterior-Superior) to NIfTI RAS.

    Args:
        dicom_slices: List of tuples (instance_number, pixel_array, header) where
            header is a pydicom.Dataset containing DICOM metadata tags

    Returns:
        4x4 affine transformation matrix mapping voxel coordinates to RAS world coordinates

    Raises:
        RuntimeError: If required DICOM tags are missing
    """
    # use the first slice to extract most parameters
    first_header = dicom_slices[0][2]

    try:
        # ImageOrientationPatient (0020,0037): direction cosines for row and column
        orientation = np.array(first_header.ImageOrientationPatient, dtype=float)
        row_cosine = orientation[:3]  # direction cosines for rows
        col_cosine = orientation[3:]  # direction cosines for columns

        # ImagePositionPatient (0020,0032): position of the upper-left voxel
        position = np.array(first_header.ImagePositionPatient, dtype=float)

        # PixelSpacing is [row_spacing, col_spacing], so map to dy, dx
        dy, dx = np.array(first_header.PixelSpacing, dtype=float)

    except AttributeError as e:
        raise RuntimeError(
            f"Missing required DICOM tag for affine calculation: {e}"
        ) from e

    # Compute Z direction and spacing (handling potential gantry tilt)
    slice_cosine = np.cross(row_cosine, col_cosine)

    # calculate slice spacing
    if len(dicom_slices) > 1:
        # calculate from the distance between first two slices
        first_pos = np.array(dicom_slices[0][2].ImagePositionPatient, dtype=float)
        second_pos = np.array(dicom_slices[1][2].ImagePositionPatient, dtype=float)
        dz = np.dot(second_pos - first_pos, slice_cosine)
    else:
        # single slice - try to get from SliceThickness or default to 1.0
        dz = float(getattr(first_header, 'SliceThickness', 1.0))

    # Construct affine in DICOM LPS space
    affine = np.eye(4)
    affine[:3, 0] = row_cosine * dx
    affine[:3, 1] = col_cosine * dy
    affine[:3, 2] = slice_cosine * dz
    affine[:3, 3] = position

    # Convert LPS to RAS by flipping X and Y axes
    return np.diag([-1, -1, 1, 1]) @ affine


def convert_dicom_to_nifti(input_path: PathLike, output_filepath: PathLike) -> None:
    """
    Convert DICOM file(s) to NIfTI format using pydicom and nibabel.

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
                header = pydicom.dcmread(dcm_file)
                # Transpose to swap (Row, Col) -> (X, Y) for NIfTI
                slices.append((header.get('InstanceNumber', 0), header.pixel_array.T, header))
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

        nib.save(nib.Nifti1Image(volume, affine), str(output_filepath))

    except Exception as e:
        raise RuntimeError(f"DICOM to NIfTI conversion failed: {e}") from e
