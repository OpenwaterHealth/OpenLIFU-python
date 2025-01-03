import shutil
from pathlib import Path

import pytest
from vtk import vtkImageData, vtkPolyData

from openlifu.db.database import Database
from openlifu.db.photoscan import (
    Photoscan,
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
    photoscan_info = example_database.get_photoscan_info(subject_id, session_id, photoscan_id)
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
