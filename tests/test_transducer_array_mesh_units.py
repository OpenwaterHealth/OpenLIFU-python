from __future__ import annotations

import copy

import numpy as np
import pytest
from vtkmodules.vtkIOGeometry import vtkOBJReader

from openlifu.db import Database
from openlifu.util.units import getunitconversion
from openlifu.xdc import TransducerArray


def _config(units="mm"):
    scale = getunitconversion("mm", units)
    return {
        "hwid": "AAA",
        "module": {"nx": 2, "ny": 1, "pitch": 20 * scale, "kerf": scale, "units": units},
    }


@pytest.mark.parametrize("mesh_field", ["registration_surface_filename", "transducer_body_filename"])
@pytest.mark.parametrize("mesh_level", ["array", "module"])
@pytest.mark.parametrize("units", ["mm", "millimeter"])
def test_inherited_mesh_matches_elements_after_database_roundtrip(tmp_path, mesh_field, mesh_level, units):
    db = Database.initialize_empty_database(tmp_path / "db")
    mesh = tmp_path / "surface.obj"
    mesh.write_text("v -10 0 0\nv 10 0 0\nv -10 1 0\nf 1 2 3\n", encoding="utf-8")
    template = TransducerArray.from_module_user_configs([_config()], arr_id="template")
    if mesh_level == "module":
        setattr(template.modules[0], mesh_field, mesh.name)
    write_kwarg = mesh_field.replace("_filename", "_model_filepath")
    db.write_transducer(template, **{write_kwarg: mesh})
    template = db.load_transducer("template", convert_array=False)
    if mesh_level == "module":
        template.attrs.pop(mesh_field)

    array = TransducerArray.from_module_user_configs([_config(units)], template=template, arr_id="device")
    db.write_transducer(array, **{write_kwarg: mesh})
    loaded = db.load_transducer("device")
    mesh_path = db.get_transducer_absolute_filepaths("device")[mesh_field.replace("_filename", "_abspath")]
    reader = vtkOBJReader()
    reader.SetFileName(mesh_path)
    reader.Update()
    vertex_mm = np.array(reader.GetOutput().GetPoint(0)) * getunitconversion(loaded.units, "mm")
    np.testing.assert_allclose(vertex_mm, loaded.get_positions(units="mm")[0])
    np.testing.assert_allclose(vertex_mm, [-10, 0, 0])

    original_template = copy.deepcopy(template.to_dict())
    with pytest.raises(ValueError, match="[Mm]esh.*units"):
        TransducerArray.from_module_user_configs([_config("cm")], template=template, arr_id="invalid")
    assert template.to_dict() == original_template
    assert "invalid" not in db.get_transducer_ids()


@pytest.mark.parametrize("mesh_field", ["registration_surface_filename", "transducer_body_filename"])
@pytest.mark.parametrize("override", [None, "", "surface.obj", "renamed.obj"])
def test_array_mesh_override_preserves_unit_guard(mesh_field, override):
    template = TransducerArray.from_module_user_configs([_config()])
    template.attrs[mesh_field] = "surface.obj"
    config = _config("cm")
    config["device"] = {"modules": [{"hwid": "AAA"}], "attrs": {mesh_field: override}}
    if override:
        with pytest.raises(ValueError, match="[Mm]esh.*units"):
            TransducerArray.from_module_user_configs([config], template=template)
    else:
        array = TransducerArray.from_module_user_configs([config], template=template)
        assert array.attrs[mesh_field] == override
        np.testing.assert_allclose(array.to_transducer().get_positions(units="mm")[0], [-10, 0, 0])


@pytest.mark.parametrize("mesh_field", ["registration_surface_filename", "transducer_body_filename"])
def test_array_mesh_requires_template_module_units(mesh_field):
    template = TransducerArray(attrs={mesh_field: "surface.obj"})
    with pytest.raises(ValueError, match="[Mm]esh.*template.*module"):
        TransducerArray.from_module_user_configs([_config()], template=template)


def test_array_mesh_uses_first_module_unit_basis():
    first, second = _config(), _config("cm")
    second["hwid"] = "BBB"
    template = TransducerArray.from_module_user_configs([first, second])
    template.attrs["transducer_body_filename"] = "body.obj"
    array = TransducerArray.from_module_user_configs([first, first], template=template)
    assert array.transducer_body_filename == "body.obj"
    with pytest.raises(ValueError, match="[Mm]esh.*units"):
        TransducerArray.from_module_user_configs([second, first], template=template)


@pytest.mark.parametrize("mesh_field", ["registration_surface_filename", "transducer_body_filename"])
@pytest.mark.parametrize("mesh_reference", [None, "", "surface.obj"])
def test_device_mesh_rejects_reordered_module_units(mesh_field, mesh_reference):
    first, second = _config(), _config("cm")
    second["hwid"] = "BBB"
    array = TransducerArray.from_module_user_configs([first, second])
    array.attrs[mesh_field] = mesh_reference
    device = array.to_device_config()
    unchanged = copy.deepcopy([first, second])
    unchanged[0]["device"] = device
    restored = TransducerArray.from_module_user_configs(unchanged)
    assert restored.attrs[mesh_field] == mesh_reference
    assert restored.to_transducer().units == "mm"

    reordered = copy.deepcopy([second, first])
    reordered[0]["device"] = device
    if mesh_reference:
        with pytest.raises(ValueError, match="[Mm]esh.*units"):
            TransducerArray.from_module_user_configs(reordered)
    else:
        restored = TransducerArray.from_module_user_configs(reordered)
        assert restored.attrs[mesh_field] == mesh_reference
        assert restored.to_transducer().units == "cm"
