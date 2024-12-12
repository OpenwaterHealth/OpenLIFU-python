
from pathlib import Path
from typing import Tuple

import numpy as np
import OpenEXR
import vtk


def ReadPolyData(file_name):

    valid_suffixes = ['.g', '.obj', '.stl', '.ply', '.vtk', '.vtp']
    path = Path(file_name)
    if path.suffix:
        ext = path.suffix.lower()
    if path.suffix not in valid_suffixes:
        raise ValueError(f"File format {path.suffix} not supported by reader")
    else:
        if ext == ".ply":
            reader = vtk.vtkPLYReader()
            reader.SetFileName(file_name)
            reader.Update()
            poly_data = reader.GetOutput()
        elif ext == ".vtp":
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(file_name)
            reader.Update()
            poly_data = reader.GetOutput()
        elif ext == ".obj":
            reader = vtk.vtkOBJReader()
            reader.SetFileName(file_name)
            reader.Update()
            poly_data = reader.GetOutput()
        elif ext == ".stl":
            reader = vtk.vtkSTLReader()
            reader.SetFileName(file_name)
            reader.Update()
            poly_data = reader.GetOutput()
        elif ext == ".vtk":
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(file_name)
            reader.Update()
            poly_data = reader.GetOutput()
        elif ext == ".g":
            reader = vtk.vtkBYUReader()
            reader.SetGeometryFileName(file_name)
            reader.Update()
            poly_data = reader.GetOutput()

        return poly_data

def convert_numpy_to_vtkimage(image_numpy):

    vtkimage_data = vtk.vtkImageData()
    vtkimage_data.SetDimensions(image_numpy.shape[1], image_numpy.shape[0], 1)
    vtkimage_data.SetNumberOfScalarComponents(image_numpy.shape[2], vtkimage_data.GetInformation())
    pd = vtkimage_data.GetPointData()
    new_rgb_data = image_numpy[::-1].reshape((-1, image_numpy.shape[2]))
    vtk_array = vtk.util.numpy_support.numpy_to_vtk(new_rgb_data, deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
    pd.SetScalars(vtk_array)
    return vtkimage_data

def ReadImageData(file_name):

    valid_suffixes = ['.jpg', '.png', '.tiff', '.exr']
    path = Path(file_name)
    if path.suffix:
        ext = path.suffix.lower()
    if path.suffix not in valid_suffixes:
        raise ValueError(f"File format {path.suffix} not supported by reader")
    else:
        if ext == ".jpg":
            reader = vtk.vtkJPEGReader()
            reader.SetFileName(file_name)
            reader.Update()
            image_data = reader.GetOutput()
        elif ext == ".png":
            reader = vtk.vtkPNGReader()
            reader.SetFileName(file_name)
            reader.Update()
            image_data = reader.GetOutput()
        elif ext == ".tiff":
            reader = vtk.vtkTIFFReader()
            reader.SetFileName(file_name)
            reader.Update()
            image_data = reader.GetOutput()
        elif ext == ".exr":
            with OpenEXR.File(file_name, separate_channels = True) as exr_file:

                R = exr_file.channels()['R'].pixels
                G = exr_file.channels()['G'].pixels
                B = exr_file.channels()['B'].pixels

                # Combine channels into a single RGB image (H x W x 3)
                rgb_data = np.stack([R, G, B], axis=-1)

                # Normalize the data to 0-255 range for compatibility with VTK
                rgb_data = np.clip(rgb_data*(2**16-1), 0, 65535)

                image_data = convert_numpy_to_vtkimage(rgb_data)

        return image_data


def load_photoscan(model_filename: Path, texture_filename: Path) -> Tuple[vtk.vtkPolyData, vtk.vtkImageData]:
    """
    Returns a tuple containing the model as a vtkPolyData and texture image as vtkImageData.
    """
    model_vtkpolydata = ReadPolyData(model_filename)
    texture_vtkimagedata = ReadImageData(texture_filename)
    return (model_vtkpolydata, texture_vtkimagedata)
