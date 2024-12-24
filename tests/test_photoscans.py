import shutil
from pathlib import Path

import pytest
from vtk import vtkImageData, vtkPolyData

from openlifu.db.database import Database
from openlifu.db.photoscans import Photoscan


@pytest.fixture()
def example_database(tmp_path:Path) -> Database:
    """Example database in a temporary directory"""
    shutil.copytree(Path(__file__).parent/'resources/example_db', tmp_path/"example_db")
    return Database(tmp_path/"example_db")

def test_load_photoscans(example_database:Database, tmp_path:Path):

    subject_id = "example_subject"
    session_id = "example_session"
    photoscan_id = "example_photoscan"
    photoscan_info = example_database.get_photoscan_info(subject_id, session_id, photoscan_id)
    photoscan = Photoscan()
    photoscan.from_filepaths(photoscan_info["model_abspath"],photoscan_info["texture_abspath"])
    assert photoscan.model is not None
    assert photoscan.texture is not None
    assert isinstance(photoscan.model, vtkPolyData)
    assert isinstance(photoscan.texture,vtkImageData)

    photoscan = Photoscan()
    photoscan.from_dict(photoscan_info)
    assert photoscan.model is not None
    assert photoscan.texture is not None
    assert isinstance(photoscan.model, vtkPolyData)
    assert isinstance(photoscan.texture,vtkImageData)

    bogus_file = Path(tmp_path/"test_db_files/bogus_photoscan.bogus")
    photoscan = Photoscan()
    with pytest.raises(ValueError, match="not supported by reader"):
        photoscan.from_filepaths(photoscan_info["model_abspath"], bogus_file)
    with pytest.raises(ValueError, match="not supported by reader"):
        photoscan.from_filepaths(bogus_file, photoscan_info["texture_abspath"])
