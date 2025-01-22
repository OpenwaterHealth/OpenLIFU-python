
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import OpenEXR
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from openlifu.util.json import PYFUSEncoder


@dataclass
class Photoscan:

    id : str = "photoscan"
    """ID of this photoscan"""

    name: Optional[str] = "Photoscan"
    """Photoscan name"""

    model_filename: Optional[str] =  None
    """Relative path to model"""

    texture_filename: Optional[str] = None
    """Relative path to texture image"""

    mtl_filename: Optional[str] = None
    """Relative path to materials file"""

    photoscan_approved: bool = False
    """Approval state of the photoscan. 'True' means means the user has provided some kind of
    confirmation that the photoscan is good enough to be used."""

    @staticmethod
    def from_json(json_string: str) -> "Photoscan":
        """Load a Photoscan from a json string"""
        return Photoscan.from_dict(json.loads(json_string))

    def to_json(self, compact: bool) -> str:
        """Serialize a Photoscan to a json string. This is different to the format written to file
        and does not contain the loaded models and texture.
        Args:
            compact:if enabled then the string is compact (not pretty). Disable for pretty.
        Returns: A json string representing the complete Photoscan object
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'), cls=PYFUSEncoder)
        else:
            return json.dumps(self.to_dict(), indent=4, cls=PYFUSEncoder)

    @staticmethod
    def from_dict(d:Dict) -> "Photoscan":
        """
        Create a Photoscan from a dictionary
        param d: Dictionary of photoscan parameters.
        returns: Photoscan object
        """
        return Photoscan(**d)

    def to_dict(self) -> Dict:
        """
        Convert the photoscan to a dictionary
        returns: Dictionary of photoscan parameters
        """
        d = self.__dict__.copy()
        return d

    @staticmethod
    def from_file(filename) -> "Photoscan":
        """
        Load a Photoscan from a metadata file
        :param filename: Name of the file
        """
        with open(filename) as f:
            return Photoscan.from_dict(json.load(f))

    def to_file(self, filename):
        """
        Save the photoscan to a file.
        :param filename: Name of the file
        """
        Path(filename).parent.parent.mkdir(exist_ok=True) #photoscan directory
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w') as file:
            file.write(self.to_json(compact = False))

def load_data_from_filepaths(model_abspath: str, texture_abspath: str) -> Tuple[vtk.vtkPolyData, vtk.vtkImageData]:
    """
    This function returns the data directly from the model and texture filepaths without requiring a photoscan object.
    param model_abspath: absolute filepath to model data
          texture abspath: absolute filepath to texture data
    Returns: Photoscan data as (model_vtkpolydata, texture_vtkimagedata)
    """
    model_polydata  = load_model(model_abspath)
    texture_imagedata = load_texture(texture_abspath)

    return (model_polydata, texture_imagedata)

def load_data_from_photoscan(photoscan: Photoscan, parent_dir) -> Tuple[vtk.vtkPolyData, vtk.vtkImageData]:
    """
    param parent_dir: parent directory containing model and texture data filepaths
    Returns: Photoscan data as (model_vtkpolydata, texture_vtkimagedata)
    """
    model_polydata  = load_model(Path(parent_dir)/photoscan.model_filename)
    texture_imagedata = load_texture(Path(parent_dir)/photoscan.texture_filename)

    return (model_polydata, texture_imagedata)

def load_model(file_name) -> vtk.vtkPolyData:
    """
    This function assumes that the model is saved to file in LPS coordinates.
    The model that is returned is in RAS coordinates
    """
    if not Path(file_name).exists():
        raise FileNotFoundError(f'Model filepath does not exist: {file_name}')
    mesh = read_as_vtkpolydata(file_name)
    mesh_ras = convert_between_ras_and_lps(mesh)

    return mesh_ras

def load_texture(file_name) -> vtk.vtkImageData:
    if not Path(file_name).exists():
        raise FileNotFoundError(f'Texture data filepath does not exist: {file_name}')
    texture = read_as_vtkimagedata(file_name)
    return texture

def read_as_vtkpolydata(file_name):

    suffix_to_reader_dict = {
        '.ply' : vtk.vtkPLYReader,
        '.vtp' : vtk.vtkXMLPolyDataReader,
        '.obj' : vtk.vtkOBJReader,
        '.stl' : vtk.vtkSTLReader,
        '.vtk' : vtk.vtkPolyDataReader,
        '.g' : vtk.vtkBYUReader
    }
    path = Path(file_name)
    ext = path.suffix.lower()
    if path.suffix not in suffix_to_reader_dict:
        raise ValueError(f"File format {ext} not supported by reader")
    reader = suffix_to_reader_dict[ext]()
    if ext == '.g':
        reader.SetGeometryName(file_name)
    else:
        reader.SetFileName(file_name)
    reader.Update()
    poly_data = reader.GetOutput()

    return poly_data

def convert_numpy_to_vtkimage(image_numpy: np.ndarray) -> vtk.vtkImageData:
    """
    Converts a numpy array with dimensions [HxWx3] representing an RGB image into vtkImageData
    """
    vtkimage_data = vtk.vtkImageData()
    vtkimage_data.SetDimensions(image_numpy.shape[1], image_numpy.shape[0], 1)
    vtkimage_data.SetNumberOfScalarComponents(image_numpy.shape[2], vtkimage_data.GetInformation())
    pd = vtkimage_data.GetPointData()
    new_rgb_data = image_numpy.reshape((-1, image_numpy.shape[2]))
    vtk_array = numpy_to_vtk(new_rgb_data, deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
    pd.SetScalars(vtk_array)
    return vtkimage_data

def read_as_vtkimagedata(file_name) -> vtk.vtkImageData:

    suffix_to_reader_dict = {
        '.jpg' : vtk.vtkJPEGReader,
        '.png' : vtk.vtkPNGReader,
        '.tiff' : vtk.vtkTIFFReader,
    }
    path = Path(file_name)
    ext = path.suffix.lower()
    if ext == '.exr':
        with OpenEXR.File(str(file_name), separate_channels = True) as exr_file:
            R = exr_file.channels()['R'].pixels
            G = exr_file.channels()['G'].pixels
            B = exr_file.channels()['B'].pixels
            # Combine channels into a single RGB image (H x W x 3)
            rgb_data = np.stack([R, G, B], axis=-1, dtype = np.float32)
            # EXR stores pixel data using half 16-bit floating point values. Convert data to an 8-bit image compatibility with VTK
            rgb_data = np.clip(rgb_data*(2**16-1), 0, 65535)
            image_data = convert_numpy_to_vtkimage(rgb_data)
    elif ext in suffix_to_reader_dict:
        reader = suffix_to_reader_dict[ext]()
        reader.SetFileName(file_name)
        reader.Update()
        image_data = reader.GetOutput()
    else:
        raise ValueError(f"File format {path.suffix} not supported by reader")

    return image_data

def convert_between_ras_and_lps(mesh : vtk.vtkPointSet) -> vtk.vtkPointSet:
    """Converts a mesh (polydata, unstructured grid or even just a point cloud) between the LPS (left-posterior-superior) coordinate system and
      RAS (right-anterior-superior) coordinate system."""

    transform_ras_to_lps = vtk.vtkTransform()
    transform_ras_to_lps.Scale(-1,-1,1)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(mesh)
    transformFilter.SetTransform(transform_ras_to_lps)
    transformFilter.Update()

    return transformFilter.GetOutput()
