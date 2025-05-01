from __future__ import annotations

from dataclasses import asdict

import pytest

from openlifu.seg.material import Material

# Mock PARAM_INFO for tests
PARAM_INFO = {
    "name": {"label": "Material name", "description": "Name for the material"},
    "sound_speed": {"label": "Sound speed (m/s)", "description": "Speed of sound in the material (m/s)"},
    "density": {"label": "Density (kg/m^3)", "description": "Mass density of the material (kg/m^3)"},
    "attenuation": {"label": "Attenuation (dB/cm/MHz)", "description": "Ultrasound attenuation in the material (dB/cm/MHz)"},
    "specific_heat": {"label": "Specific heat (J/kg/K)", "description": "Specific heat capacity of the material (J/kg/K)"},
    "thermal_conductivity": {"label": "Thermal conductivity (W/m/K)", "description": "Thermal conductivity of the material (W/m/K)"},
}

@pytest.fixture()
def default_material():
    return Material()

def test_default_material_values(default_material):
    assert default_material.name == "Material"
    assert default_material.sound_speed == 1500.0
    assert default_material.density == 1000.0
    assert default_material.attenuation == 0.0
    assert default_material.specific_heat == 4182.0
    assert default_material.thermal_conductivity == 0.598

def test_material_to_dict(default_material):
    expected = asdict(default_material)
    assert default_material.to_dict() == expected

def test_material_from_dict():
    data = {
        "name": "test",
        "sound_speed": 1234.0,
        "density": 999.0,
        "attenuation": 1.2,
        "specific_heat": 4000.0,
        "thermal_conductivity": 0.5
    }
    material = Material.from_dict(data)
    assert material.to_dict() == data

def test_get_param_valid(default_material):
    assert default_material.get_param("density") == 1000.0

def test_get_param_invalid(default_material):
    with pytest.raises(ValueError, match="Parameter fake_param not found."):
        default_material.get_param("fake_param")

def test_param_info_valid(monkeypatch):
    monkeypatch.setattr("openlifu.seg.material.PARAM_INFO", PARAM_INFO)
    info = Material.param_info("density")
    assert info["label"] == "Density (kg/m^3)"

def test_param_info_invalid(monkeypatch):
    monkeypatch.setattr("openlifu.seg.material.PARAM_INFO", PARAM_INFO)
    with pytest.raises(ValueError, match="Parameter unknown not found."):
        Material.param_info("unknown")
