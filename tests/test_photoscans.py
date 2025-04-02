from __future__ import annotations

import importlib.resources
import shutil
from pathlib import Path

import numpy as np
import pytest
from vtk import VTK_UNSIGNED_SHORT, vtkImageData, vtkPoints, vtkPolyData

from openlifu.db.database import Database
from openlifu.photoscan import (
    Photoscan,
    convert_between_ras_and_lps,
    convert_numpy_to_vtkimage,
    load_data_from_filepaths,
    load_data_from_photoscan,
)


@pytest.fixture()
def example_database(tmp_path:Path) -> Database:
    """Example database in a temporary directory"""
    shutil.copytree(Path(__file__).parent/'resources/example_db', tmp_path/"example_db")
    return Database(tmp_path/"example_db")

@pytest.fixture()
def example_photoscan() -> Photoscan:
    return Photoscan.from_file(Path(__file__).parent/'resources/example_db/subjects/example_subject/sessions/example_session/photoscans/example_photoscan/example_photoscan.json')

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_photoscan(example_photoscan : Photoscan, compact_representation: bool):
    reconstructed_photoscan =  example_photoscan.from_json(example_photoscan.to_json(compact_representation))
    assert example_photoscan == reconstructed_photoscan

def test_load_photoscan(example_database:Database, tmp_path:Path):

    subject_id = "example_subject"
    session_id = "example_session"
    photoscan_id = "example_photoscan"
    photoscan_info = example_database.get_photoscan_absolute_filepaths_info(subject_id, session_id, photoscan_id)
    [model, texture] = load_data_from_filepaths(photoscan_info["model_abspath"], photoscan_info["texture_abspath"])
    assert model is not None
    assert texture is not None
    assert isinstance(model, vtkPolyData)
    assert isinstance(texture,vtkImageData)

    # From photoscan object
    photoscan_metadata_filepath = example_database.get_photoscan_metadata_filepath(subject_id, session_id, photoscan_id)
    photoscan = Photoscan.from_file(photoscan_metadata_filepath)
    [model, texture] = load_data_from_photoscan(photoscan, parent_dir = Path(photoscan_metadata_filepath).parent)
    assert model is not None
    assert texture is not None
    assert isinstance(model, vtkPolyData)
    assert isinstance(texture,vtkImageData)

    bogus_file = Path(tmp_path/"test_db_files/bogus_photoscan.pdf")
    bogus_file.parent.mkdir(parents=True, exist_ok=True)
    bogus_file.touch()
    with pytest.raises(ValueError, match="not supported by reader"):
        load_data_from_filepaths(photoscan_info["model_abspath"], bogus_file)
    with pytest.raises(ValueError, match="not supported by reader"):
        load_data_from_filepaths(bogus_file,photoscan_info["texture_abspath"])

    # File does not exist
    bogus_file = Path(tmp_path/"test_db_files/bogus_photoscan.obj")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_data_from_filepaths(bogus_file, photoscan_info["texture_abspath"])

def test_convert_between_ras_and_lps():

    # Create example data
    examplePointSet = vtkPolyData()
    points = vtkPoints()

    # Add known points in LPS coordinate space
    points.InsertNextPoint(10, 20, 30)  # Example point in LPS
    points.InsertNextPoint(-15, 25, -35)
    points.InsertNextPoint(0, 0, 0)
    examplePointSet.SetPoints(points)

    expected_ras_points = [
        (-10, -20, 30),  # Flip the sign of X and Y coordinates
        (15, -25, -35),
        (0, 0, 0)
        ]

    convertedPointSet = convert_between_ras_and_lps(examplePointSet)
    converted_points = convertedPointSet.GetPoints()
    for i in range(converted_points.GetNumberOfPoints()):
        # Compare each transformed point with expected RAS points
        converted_point = np.array(convertedPointSet.GetPoint(i))
        expected_point = np.array(expected_ras_points[i])
        np.testing.assert_array_almost_equal(converted_point, expected_point)

def test_convert_numpy_to_vtkimage():

    # Create a dummy 3x2 RGB image as a numpy array with dtype uint16
    dummy_image = np.array(
        [[[65535, 0, 0], [0, 65535, 0]],
         [[0, 0, 65535], [65535, 65535, 65535]],
         [[100, 200, 300], [400, 500, 600]],],
        dtype=np.uint16
    )

    # Convert the numpy array to vtkImageData
    vtk_image = convert_numpy_to_vtkimage(dummy_image)

    # Check dimensions
    assert(vtk_image.GetDimensions() == (dummy_image.shape[1], dummy_image.shape[0],1))

    # Check scalar type and number of components
    vtk_array = vtk_image.GetPointData().GetScalars()
    assert(vtk_array.GetNumberOfComponents() == 3)
    assert(vtk_array.GetDataType() == VTK_UNSIGNED_SHORT)

    # Probe specific coordinates and compare values
    for i in range(dummy_image.shape[0]):
        for j in range(dummy_image.shape[1]):
            # Get the expected RGB values from the numpy array
            expected_rgb = dummy_image[i, j]

            # Compute the flat index of the pixel in vtkImageData
            flat_index = i * dummy_image.shape[1] + j

            # Get the RGB values from vtkImageData
            vtk_rgb = [
                vtk_array.GetComponent(flat_index, 0),
                vtk_array.GetComponent(flat_index, 1),
                vtk_array.GetComponent(flat_index, 2),
            ]

            # Assert that the values are equal
            assert(list(vtk_rgb) ==  list(expected_rgb))

def test_resource_import():
    """Ensure that a meshroom pipeline resource file can be imported"""
    pipeline_path = importlib.resources.files("openlifu.meshroom_pipelines") / "default_pipeline.mg"
    assert pipeline_path.exists()
