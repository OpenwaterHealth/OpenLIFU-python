
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import OpenEXR
import vtk
import vtk.util


def read_as_vtkpolydata(file_name):

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
    new_rgb_data = image_numpy.reshape((-1, image_numpy.shape[2]))
    #new_rgb_data = np.flipud(new_rgb_data) # To look like blender format upon loading. Image is instead flipped in texture module.
    vtk_array = vtk.util.numpy_support.numpy_to_vtk(new_rgb_data, deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
    pd.SetScalars(vtk_array)
    return vtkimage_data

def read_as_vtkimagedata(file_name):

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
            with OpenEXR.File(str(file_name), separate_channels = True) as exr_file:
                R = exr_file.channels()['R'].pixels
                G = exr_file.channels()['G'].pixels
                B = exr_file.channels()['B'].pixels
                # Combine channels into a single RGB image (H x W x 3)
                rgb_data = np.stack([R, G, B], axis=-1)
                # Normalize the data to 0-255 range for compatibility with VTK
                rgb_data = np.clip(rgb_data*(2**16-1), 0, 65535)
                image_data = convert_numpy_to_vtkimage(rgb_data)

        return image_data

def convert_between_ras_and_lps(mesh):

    transform_ras_to_lps = vtk.vtkTransform()
    transform_ras_to_lps.Scale(-1,-1,1)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(mesh)
    transformFilter.SetTransform(transform_ras_to_lps)
    transformFilter.Update()

    return transformFilter.GetOutput()

def load_model(file_name):

    mesh = read_as_vtkpolydata(file_name)
    mesh_ras = convert_between_ras_and_lps(mesh)

    return mesh_ras

@dataclass
class Photoscan:

    id : Optional[str] = "photoscan"
    """ID of this photoscan"""

    name: Optional[str] = "Photoscan"
    """Photoscan name"""

    model_abspath: str =  ""
    """ Absolute path to model"""

    texture_abspath: str = ""
    """Absolute path to texture image"""

    mtl_abspath: Optional[str] = ""
    """Absolute path to materials file"""

    model: vtk.vtkPolyData = None
    """Loaded model"""

    texture: vtk.vtkImageData = None
    """Loaded texture image"""

    approved: bool = False
    """Approval state of the photoscan. 'True' means means the user has provided some kind of
    confirmation that the photoscan is good enough to be used."""

    @staticmethod
    def from_dict(d:Dict):
        """
        Create a Photoscan from a dictionary
        param d: Dictionary of photoscan parameters.
        returns: Photoscan object
        """
        if 'model_abspath' in d:
            d['model'] = load_model(d['model_abspath'])
        if 'texture_abspath' in d:
            d['texture'] = read_as_vtkimagedata(d['texture_abspath'])

        return Photoscan(**d)

    def to_dict(self):
        """
        Convert the photoscan to a dictionary"

        : returns: Dictionary of photoscan parameters
        """
        d = self.__dict__.copy()
        return d

    def from_filepaths(self, model_abspath: str, texture_abspath: str):
        """
        params: absolute filepath to model and texture files
        return: Creates a photoscan containing the loaded model and texture data
        """
        d = {'model_abspath': model_abspath, 'texture_abspath': texture_abspath}
        d['model'] = load_model(model_abspath)
        d['texture'] = read_as_vtkimagedata(texture_abspath)
        return Photoscan(**d)
