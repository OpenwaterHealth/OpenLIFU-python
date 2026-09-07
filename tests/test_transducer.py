from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from helpers import dataclasses_are_equal

from openlifu.xdc import DeviceConfigMismatchError, Element, Transducer, TransducerArray
from openlifu.xdc.transducerarray import (
    _build_meshless_default_template,
    get_angle_from_gap,
    get_gap_from_angle,
    get_roc_from_angle,
)


@pytest.fixture()
def example_transducer() -> Transducer:
    return Transducer.from_file(Path(__file__).parent/'resources/example_db/transducers/example_transducer/example_transducer.json')

def load_transducer_array(transducer_array_id : str) -> TransducerArray:
    """Load an example TransducerArray given the transducer ID."""
    return TransducerArray.from_file(Path(__file__).parent/f'resources/example_db/transducers/{transducer_array_id}/{transducer_array_id}.json')

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_transducer(example_transducer : Transducer, compact_representation: bool):
    reconstructed_transducer = example_transducer.from_json(example_transducer.to_json(compact_representation))
    dataclasses_are_equal(example_transducer, reconstructed_transducer)

def test_get_polydata_color_options(example_transducer : Transducer):
    """Ensure that the color is set correctly on the polydata"""
    polydata_with_default_color = example_transducer.get_polydata()
    point_scalars = polydata_with_default_color.GetPointData().GetScalars()
    assert point_scalars is None

    polydata_with_given_color = example_transducer.get_polydata(facecolor=[0,1,1,0.5])
    point_scalars = polydata_with_given_color.GetPointData().GetScalars()
    assert point_scalars is not None

def test_default_transducer():
    """Ensure it is possible to construct a default transducer"""
    Transducer()

def test_convert_transform():
    transducer = Transducer(units='cm')
    transform = transducer.convert_transform(
        matrix = np.array([
            [1,0,0,2],
            [0,1,0,3],
            [0,0,1,4],
            [0,0,0,1],
        ], dtype=float),
        units = "m",
    )
    expected_transform = np.array([
        [1,0,0,200],
        [0,1,0,300],
        [0,0,1,400],
        [0,0,0,1],
    ], dtype=float)
    assert np.allclose(transform,expected_transform)

def test_get_effective_origin():
    transducer = Transducer.gen_matrix_array(nx=3, ny=2, units='cm')
    effective_origin_with_all_active = transducer.get_effective_origin(apodizations = np.ones(transducer.numelements()))
    assert np.allclose(effective_origin_with_all_active, np.zeros(3))

    rng = np.random.default_rng()
    element_index_to_turn_on = rng.integers(transducer.numelements())
    apodizations_with_just_one_element = np.zeros(transducer.numelements())
    apodizations_with_just_one_element[element_index_to_turn_on] = 0.5 # It is allowed to be a number between 0 and 1
    assert np.allclose(
        transducer.get_effective_origin(apodizations = apodizations_with_just_one_element, units = "um"),
        transducer.get_positions(units="um")[element_index_to_turn_on],
    )

def test_get_standoff_transform_in_units():
    standoff_transform_in_mm = np.array([
            [-0.1,0.9,0,20],
            [0.9,0.1,0,30],
            [0,0,1,40],
            [0,0,0,1],
    ])
    standoff_transform_in_cm = np.array([
            [-0.1,0.9,0,2],
            [0.9,0.1,0,3],
            [0,0,1,4],
            [0,0,0,1],
    ])
    transducer = Transducer(units='mm')
    transducer.standoff_transform = standoff_transform_in_mm
    assert np.allclose(
        transducer.get_standoff_transform_in_units("cm"),
        standoff_transform_in_cm,
    )

def test_read_data_types(example_transducer:Transducer):
    assert isinstance(example_transducer.standoff_transform, np.ndarray)
    if len(example_transducer.elements) > 0:
        assert isinstance(example_transducer.elements[0], Element)

@pytest.mark.parametrize(
    "transducer_array_id",
    [
        "example_transducer_array",
        "example_transducer_array2",
    ]
)
def test_transducer_array_to_transducer_data_types(transducer_array_id):
    transducer_array : TransducerArray = load_transducer_array(transducer_array_id)
    transducer = transducer_array.to_transducer()
    assert isinstance(transducer.standoff_transform, np.ndarray)
    assert not hasattr(transducer, "impulse_response")
    assert not hasattr(transducer, "impulse_dt")
    if len(transducer.elements) > 0:
        assert isinstance(transducer.elements[0], Element)


def test_transducer_calc_output_interpolates_dictionary_sensitivity():
    transducer = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 1.0), (300e3, 3.0)],
    )
    transducer.elements[0].sensitivity = 1.0
    cycles = 3
    dt = 1e-7

    output_mid = transducer.calc_output(cycles=cycles, frequency=200e3, dt=dt)
    output_low = transducer.calc_output(cycles=cycles, frequency=100e3, dt=dt)

    n_samples_mid = int(np.round(cycles / (200e3 * dt)))
    n_samples_low = int(np.round(cycles / (100e3 * dt)))
    t_mid = np.arange(n_samples_mid) * dt
    t_low = np.arange(n_samples_low) * dt
    expected_mid = 2.0 * np.sin(2 * np.pi * 200e3 * t_mid)
    expected_low = 1.0 * np.sin(2 * np.pi * 100e3 * t_low)

    np.testing.assert_allclose(output_mid[0], expected_mid)
    np.testing.assert_allclose(output_low[0], expected_low)


def test_element_calc_output_generates_signal_from_scalar_input():
    element = Element(sensitivity=2.0)
    cycles = 4
    frequency = 100e3
    dt = 1e-7
    n_samples = int(np.round(cycles / (frequency * dt)))

    output = element.calc_output(cycles=cycles, frequency=frequency, dt=dt, amplitude=3.0)
    t = np.arange(n_samples) * dt
    expected = 2.0 * 3.0 * np.sin(2 * np.pi * frequency * t)

    np.testing.assert_allclose(output, expected)


def test_element_calc_output_enforces_cycles_duration_for_generated_signal():
    element = Element(sensitivity=1.0)
    cycles = 1
    frequency = 200e3
    dt = 1e-6
    n_samples = int(np.round(cycles / (frequency * dt)))
    output = element.calc_output(cycles=cycles, frequency=frequency, dt=dt)
    t = np.arange(n_samples) * dt
    expected = np.sin(2 * np.pi * frequency * t)

    assert len(output) == n_samples
    np.testing.assert_allclose(output, expected)


def test_merge_pushes_transducer_sensitivity_into_elements():
    transducer_a = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 2.0), (300e3, 4.0)],
    )
    transducer_b = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 3.0), (300e3, 6.0)],
    )
    transducer_a.elements[0].sensitivity = 5.0
    transducer_b.elements[0].sensitivity = 7.0

    merged = Transducer.merge([transducer_a, transducer_b], merge_mismatched_sensitivity=True)

    assert merged.sensitivity == 1.0
    assert merged.elements[0].sensitivity == [(100e3, 10.0),(300e3, 20.0)]
    assert merged.elements[1].sensitivity == [(100e3, 21.0),(300e3, 42.0)]


def test_merge_rejects_mismatched_sensitivity_keys():
    transducer_a = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 2.0), (300e3, 4.0)],
    )
    transducer_b = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 2.0), (300e3, 4.0)],
    )
    transducer_a.elements[0].sensitivity = [(100e3, 5.0), (300e3, 7.0)]
    transducer_b.elements[0].sensitivity = [(100e3, 11.0), (400e3, 13.0)]

    with pytest.raises(ValueError, match="different frequency keys"):
        Transducer.merge([transducer_a, transducer_b], merge_mismatched_sensitivity=True)


@pytest.mark.parametrize(
    ("width", "dth", "roc"),
    [
        (8.0, 0.08, 25.0),
        (10.0, 0.12, 30.0),
        (12.0, 0.18, 45.0),
    ],
)
def test_concave_geometry_helpers_are_mutual_inverses(width: float, dth: float, roc: float):
    gap = get_gap_from_angle(width, dth, roc)
    recovered_roc = get_roc_from_angle(width, gap, dth)
    recovered_dth = get_angle_from_gap(width, gap, roc)
    recovered_gap = get_gap_from_angle(width, recovered_dth, roc)

    assert np.isclose(recovered_roc, roc)
    assert np.isclose(recovered_dth, dth)
    assert np.isclose(recovered_gap, gap)


def test_get_concave_cylinder_computes_gap_from_dth_and_roc_layout_spacing():
    base = Transducer.gen_matrix_array(nx=1, ny=1, units="mm")
    width = 8.0
    dth = 0.12
    roc = 25.0
    array = TransducerArray.get_concave_cylinder(
        base,
        rows=2,
        cols=1,
        width=width,
        dth=dth,
        roc=roc,
        units="mm",
    )
    merged = array.to_transducer()
    positions = merged.get_positions(units="mm")

    expected_gap = get_gap_from_angle(width, dth, roc)
    y_spacing = np.abs(positions[1, 1] - positions[0, 1])

    assert np.isclose(y_spacing, width + expected_gap)


def test_get_concave_cylinder_handles_zero_dth_without_roc():
    base = Transducer.gen_matrix_array(nx=1, ny=1, units="mm")
    width = 10.0
    gap = 2.0
    array = TransducerArray.get_concave_cylinder(
        base,
        rows=1,
        cols=2,
        width=width,
        gap=gap,
        dth=0.0,
        units="mm",
    )
    merged = array.to_transducer()
    positions = merged.get_positions(units="mm")

    x_spacing = np.abs(positions[1, 0] - positions[0, 0])
    z_values = positions[:, 2]

    assert np.isclose(x_spacing, width + gap)
    np.testing.assert_allclose(z_values, np.zeros_like(z_values))


def test_get_concave_cylinder_rejects_gap_dth_roc_together():
    base = Transducer.gen_matrix_array(nx=1, ny=1, units="mm")
    with pytest.raises(ValueError, match="cannot specify all of gap, dth, and roc"):
        TransducerArray.get_concave_cylinder(
            base,
            rows=1,
            cols=2,
            width=10.0,
            gap=1.0,
            dth=0.2,
            roc=20.0,
            units="mm",
        )


def test_transducer_calc_output_combines_frequency_dependent_sensitivities():
    transducer = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 2.0), (300e3, 4.0)],
    )
    transducer.elements[0].sensitivity = [(100e3, 5.0), (300e3, 9.0)]

    frequency = 200e3
    dt = 1e-7
    cycles = 3
    n_samples = int(np.round(cycles / (frequency * dt)))
    t = np.arange(n_samples) * dt
    expected_drive = np.sin(2 * np.pi * frequency * t)

    output = transducer.calc_output(cycles=cycles, frequency=frequency, dt=dt)

    np.testing.assert_allclose(output[0], 21.0 * expected_drive)


def test_transducer_array_to_transducer_preserves_frequency_dependent_sensitivities():
    transducer_a = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 2.0), (300e3, 4.0)],
    )
    transducer_b = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 1.0), (300e3, 3.0)],
    )
    transducer_a.elements[0].sensitivity = 5.0
    transducer_b.elements[0].sensitivity = 7.0

    array = TransducerArray.get_concave_cylinder(
        [transducer_a, transducer_b],
        rows=1,
        cols=2,
        width=10.0,
        gap=0.0,
        units="mm",
    )
    merged = array.to_transducer()

    frequency = 200e3
    dt = 1e-7
    cycles = 2
    n_samples = int(np.round(cycles / (frequency * dt)))
    t = np.arange(n_samples) * dt
    expected_drive = np.sin(2 * np.pi * frequency * t)

    output = merged.calc_output(cycles=cycles, frequency=frequency, dt=dt)

    np.testing.assert_allclose(output[0], 15.0 * expected_drive)
    np.testing.assert_allclose(output[1], 14.0 * expected_drive)


def test_element_sensitivity_from_json_is_list_of_tuples():
    """Sensitivity read from a JSON dict (list-of-lists) is converted to List[tuple[float, float]]."""
    d = {
        "index": 1,
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0],
        "size": [1.0, 1.0],
        "pin": 1,
        "units": "mm",
        "sensitivity": [[100e3, 1.0], [300e3, 3.0]],  # JSON encodes tuples as lists
    }
    element = Element.from_dict(d)
    assert isinstance(element.sensitivity, list)
    assert all(isinstance(pair, tuple) for pair in element.sensitivity)
    assert all(isinstance(f, float) and isinstance(v, float) for f, v in element.sensitivity)
    assert element.sensitivity == [(100e3, 1.0), (300e3, 3.0)]


def test_transducer_sensitivity_from_json_is_list_of_tuples():
    """Transducer-level sensitivity survives a to_json/from_json round-trip as List[tuple[float, float]]."""
    transducer = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 2.0), (300e3, 4.0)],
    )
    reconstructed = Transducer.from_json(transducer.to_json())
    assert isinstance(reconstructed.sensitivity, list)
    assert all(isinstance(pair, tuple) for pair in reconstructed.sensitivity)
    assert all(isinstance(f, float) and isinstance(v, float) for f, v in reconstructed.sensitivity)
    assert reconstructed.sensitivity == [(100e3, 2.0), (300e3, 4.0)]


def test_element_in_transducer_sensitivity_from_json_is_list_of_tuples():
    """Element-level sensitivity inside a Transducer survives a to_json/from_json round-trip as List[tuple[float, float]]."""
    transducer = Transducer.gen_matrix_array(nx=1, ny=1, units="mm")
    transducer.elements[0].sensitivity = [(100e3, 5.0), (300e3, 9.0)]
    reconstructed = Transducer.from_json(transducer.to_json())
    el_sensitivity = reconstructed.elements[0].sensitivity
    assert isinstance(el_sensitivity, list)
    assert all(isinstance(pair, tuple) for pair in el_sensitivity)
    assert all(isinstance(f, float) and isinstance(v, float) for f, v in el_sensitivity)
    assert el_sensitivity == [(100e3, 5.0), (300e3, 9.0)]


def _example_module_user_config(hwid: str = "ABCD1234") -> dict:
    return {
        "sn": "EVT2B-400K-TEST",
        "hwid": hwid,
        "freq": 400,
        "module": {
            "id": f"txm_400_{hwid.lower()}",
            "name": f"TXM 400kHz ({hwid})",
            "nx": 8,
            "ny": 8,
            "pitch": 5,
            "frequency": 400000.0,
            "kerf": 0.3,
            "crosstalk_frac": 0.12,
            "crosstalk_dist": 0.00505,
            "sensitivity": [(400e3, 2800.0), (405e3, 1950.0)],
        },
        "device": {},
    }


def test_transducer_from_module_user_config():
    cfg = _example_module_user_config(hwid="HW1")
    t = Transducer.from_module_user_config(cfg)
    assert isinstance(t, Transducer)
    assert t.numelements() == 64
    assert t.id == "txm_400_hw1"
    assert t.frequency == 400000.0
    assert t.attrs["hwid"] == "HW1"
    assert t.sensitivity == [(400e3, 2800.0), (405e3, 1950.0)]


def test_transducer_from_module_user_config_missing_module():
    with pytest.raises(ValueError, match="no 'module'"):
        Transducer.from_module_user_config({"hwid": "X"})


def test_transducer_array_from_module_user_configs_bare():
    cfgs = [_example_module_user_config("HW1"), _example_module_user_config("HW2")]
    arr = TransducerArray.from_module_user_configs(cfgs)
    assert isinstance(arr, TransducerArray)
    assert len(arr.modules) == 2
    assert arr.id == "transducer_array"
    for m in arr.modules:
        np.testing.assert_allclose(m.transform, np.eye(4))
    assert {m.attrs.get("hwid") for m in arr.modules} == {"HW1", "HW2"}


def test_transducer_array_from_module_user_configs_with_device_field():
    cfg1 = _example_module_user_config("HW1")
    cfg2 = _example_module_user_config("HW2")
    cfg1["device"] = {
        "id": "test_array",
        "name": "Test Array",
        "modules": [
            {"hwid": "HW2", "transform": np.diag([1, 1, 1, 1]).tolist()},
            {"hwid": "HW1",
             "transform": [[1, 0, 0, 10.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        ],
        "attrs": {"registration_surface_filename": "x.obj"},
    }
    arr = TransducerArray.from_module_user_configs([cfg1, cfg2])
    assert arr.id == "test_array"
    assert arr.name == "Test Array"
    assert arr.attrs["registration_surface_filename"] == "x.obj"
    np.testing.assert_allclose(arr.modules[0].transform[0, 3], 10.0)
    np.testing.assert_allclose(arr.modules[1].transform, np.eye(4))


def test_transducer_array_from_module_user_configs_with_template():
    cfgs = [_example_module_user_config("HW1"), _example_module_user_config("HW2")]
    base_template = TransducerArray.get_concave_cylinder(
        Transducer.gen_matrix_array(nx=8, ny=8, pitch=5, kerf=0.3, units="mm"),
        rows=1, cols=2, width=40, gap=0.0, units="mm",
        id="template_array", name="Template Array",
        attrs={"registration_surface_filename": "tpl.obj"},
    )
    for m in base_template.modules:
        m.registration_surface_filename = "module.surf.obj"
        m.transducer_body_filename = "module.body.obj"

    arr = TransducerArray.from_module_user_configs(cfgs, template=base_template)
    assert arr.id == "template_array"
    assert arr.attrs["registration_surface_filename"] == "tpl.obj"
    for m in arr.modules:
        assert m.registration_surface_filename == "module.surf.obj"
        assert m.transducer_body_filename == "module.body.obj"
    np.testing.assert_allclose(arr.modules[0].transform, base_template.modules[0].transform)


def test_transducer_array_from_module_user_configs_module_transforms_override():
    cfgs = [_example_module_user_config("HW1"), _example_module_user_config("HW2")]
    cfgs[0]["device"] = {
        "id": "x",
        "name": "x",
        "modules": [
            {"hwid": "HW1", "transform": np.eye(4).tolist()},
            {"hwid": "HW2", "transform": np.eye(4).tolist()},
        ],
        "attrs": {},
    }
    overrides = [
        np.array([[1, 0, 0, 1.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
        np.array([[1, 0, 0, 2.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
    ]
    arr = TransducerArray.from_module_user_configs(cfgs, module_transforms=overrides)
    np.testing.assert_allclose(arr.modules[0].transform[0, 3], 1.0)
    np.testing.assert_allclose(arr.modules[1].transform[0, 3], 2.0)


def test_transducer_array_from_module_user_configs_empty_raises():
    with pytest.raises(ValueError, match="at least one user_config"):
        TransducerArray.from_module_user_configs([])


def test_transducer_array_from_module_user_configs_length_mismatch_raises():
    cfgs = [_example_module_user_config("HW1")]
    with pytest.raises(ValueError, match="module_transforms length"):
        TransducerArray.from_module_user_configs(cfgs, module_transforms=[np.eye(4), np.eye(4)])


def test_transducer_array_from_module_user_configs_explicit_arr_id_name_override():
    cfg1 = _example_module_user_config("HW1")
    cfg2 = _example_module_user_config("HW2")
    cfg1["device"] = {
        "id": "from_device",
        "name": "From Device",
        "modules": [
            {"hwid": "HW1", "transform": np.eye(4).tolist()},
            {"hwid": "HW2", "transform": np.eye(4).tolist()},
        ],
        "attrs": {},
    }
    arr = TransducerArray.from_module_user_configs(
        [cfg1, cfg2], arr_id="explicit_id", arr_name="Explicit Name",
    )
    assert arr.id == "explicit_id"
    assert arr.name == "Explicit Name"


def test_transducer_array_from_module_user_configs_arr_id_falls_through():
    cfgs = [_example_module_user_config("HW1"), _example_module_user_config("HW2")]
    template = TransducerArray.get_concave_cylinder(
        Transducer.gen_matrix_array(nx=8, ny=8, pitch=5, kerf=0.3, units="mm"),
        rows=1, cols=2, width=40, gap=0.0, units="mm",
        id="tpl_id", name="Tpl Name",
    )
    arr = TransducerArray.from_module_user_configs(cfgs, template=template)
    assert arr.id == "tpl_id"
    assert arr.name == "Tpl Name"


@pytest.mark.parametrize("module", [None, {}, [], [1], "module"])
def test_transducer_from_module_user_config_requires_nonempty_dict(module):
    with pytest.raises(ValueError, match="no 'module'"):
        Transducer.from_module_user_config({"module": module})


def test_transducer_from_module_user_config_geometry_and_independence():
    cfg = _example_module_user_config("HW1")
    cfg["module"].update({
        "nx": 3,
        "ny": 2,
        "pitch": 2,
        "kerf": 0.5,
        "units": "cm",
        "attrs": {"calibration": {"values": [1, 2]}, "hwid": "OLD"},
        "module_invert": [True],
    })
    original = copy.deepcopy(cfg)

    transducer = Transducer.from_module_user_config(cfg)

    assert transducer.numelements() == 6
    assert transducer.units == "cm"
    assert transducer.name == cfg["module"]["name"]
    assert transducer.frequency == 400e3
    assert transducer.crosstalk_frac == 0.12
    assert transducer.crosstalk_dist == 0.00505
    assert transducer.attrs["hwid"] == "HW1"
    np.testing.assert_allclose(transducer.elements[0].get_position(), [-2, 1, 0])
    np.testing.assert_allclose(transducer.elements[-1].get_position(), [2, -1, 0])
    np.testing.assert_allclose(transducer.elements[0].get_size(), [1.5, 1.5])
    assert [el.pin for el in transducer.elements] == list(range(1, 7))
    assert [el.index for el in transducer.elements] == list(range(1, 7))
    assert transducer.registration_surface_filename is None
    assert transducer.transducer_body_filename is None
    np.testing.assert_array_equal(transducer.standoff_transform, np.eye(4))
    transducer.to_json()
    transducer.attrs["calibration"]["values"].append(3)
    transducer.module_invert[0] = False
    assert cfg == original
    cfg["module"]["sensitivity"].append((410e3, 1000.0))
    assert transducer.sensitivity == original["module"]["sensitivity"]


def test_transducer_from_module_user_config_without_hwid():
    cfg = _example_module_user_config()
    cfg.pop("hwid")
    assert "hwid" not in Transducer.from_module_user_config(cfg).attrs


def _translation(x):
    transform = np.eye(4)
    transform[0, 3] = x
    return transform


def test_transducer_array_from_module_user_configs_precedence_and_independence():
    cfgs = [_example_module_user_config("HW1"), _example_module_user_config("HW2")]
    cfgs[0]["module"]["attrs"] = {"calibration": {"values": [1, 2]}}
    template = TransducerArray.get_concave_cylinder(
        Transducer.gen_matrix_array(nx=1, ny=1, frequency=155e3, sensitivity=10),
        cols=2, id="template", name="Template",
        attrs={
            "registration_surface_filename": "template.obj",
            "retained": {"values": [1]},
            "overridden": "template",
            "standoff_transform": _translation(3),
        },
    )
    for i, module in enumerate(template.modules):
        module.registration_surface_filename = f"module{i}.surface.obj"
        module.transducer_body_filename = f"module{i}.body.obj"
        module.standoff_transform = _translation(i + 1)
        module.module_invert = [True]
    cfgs[0]["device"] = {
        "id": "device", "name": "Device",
        "attrs": {"overridden": {"values": [2]}, "standoff_transform": _translation(7).tolist()},
        "modules": [
            {"hwid": "HW2", "transform": _translation(20).tolist()},
            {"hwid": "HW1", "transform": _translation(10).tolist()},
        ],
    }
    cfgs[1]["device"] = {"id": "ignored", "name": "Ignored"}
    original_cfgs = copy.deepcopy(cfgs)
    original_template = copy.deepcopy(template)
    overrides = [_translation(100), _translation(200)]

    array = TransducerArray.from_module_user_configs(cfgs, template=template)
    explicit = TransducerArray.from_module_user_configs(
        cfgs, template=template, module_transforms=overrides,
        arr_id="explicit", arr_name="Explicit",
    )

    assert (array.id, array.name) == ("device", "Device")
    assert (explicit.id, explicit.name) == ("explicit", "Explicit")
    assert array.attrs["registration_surface_filename"] == "template.obj"
    assert array.attrs["overridden"] == {"values": [2]}
    np.testing.assert_array_equal(array.attrs["standoff_transform"], _translation(7))
    for i, module in enumerate(array.modules):
        assert module.numelements() == 64
        assert module.frequency == 400e3
        assert module.sensitivity == [(400e3, 2800.0), (405e3, 1950.0)]
        assert module.crosstalk_frac == 0.12
        assert module.crosstalk_dist == 0.00505
        assert module.attrs["hwid"] == f"HW{i + 1}"
        assert module.registration_surface_filename == f"module{i}.surface.obj"
        assert module.transducer_body_filename == f"module{i}.body.obj"
        assert module.module_invert == [True]
        np.testing.assert_array_equal(module.standoff_transform, _translation(i + 1))
        np.testing.assert_array_equal(module.transform, _translation((i + 1) * 10))
        np.testing.assert_array_equal(explicit.modules[i].transform, overrides[i])

    array.to_json()
    array.attrs["retained"]["values"].append(9)
    array.attrs["overridden"]["values"].append(9)
    array.attrs["standoff_transform"][0, 3] = 9
    array.modules[0].attrs["calibration"]["values"].append(9)
    array.modules[0].module_invert[0] = False
    array.modules[0].standoff_transform[0, 3] = 9
    array.modules[0].transform[0, 3] = 9
    explicit.modules[0].transform[0, 3] = 9
    assert cfgs == original_cfgs
    assert dataclasses_are_equal(template, original_template)
    np.testing.assert_array_equal(overrides[0], _translation(100))


def test_transducer_array_device_transforms_use_position_without_hwids():
    cfgs = [_example_module_user_config("HW1"), _example_module_user_config("HW2")]
    cfgs[0]["device"] = {"modules": [
        {"transform": _translation(1).tolist()},
        {"transform": _translation(2).tolist()},
    ]}
    array = TransducerArray.from_module_user_configs(cfgs)
    for i, module in enumerate(array.modules):
        np.testing.assert_array_equal(module.transform, _translation(i + 1))


@pytest.mark.parametrize(
    ("recorded_hwids", "connected_hwids", "valid"),
    [
        (["HW1", "HW2"], ["HW1", "HW2"], True),
        (["HW2", "HW1"], ["HW1", "HW2"], True),
        (["HW1"], ["HW1", "HW2"], False),
        (["HW1", "HW2"], ["HW1", "OTHER"], False),
        (["HW1", "HW2"], ["HW1", None], False),
        ([None, None], ["HW1", "HW2"], True),
        ([None, None], [None, None], True),
        (["HW1", None], ["HW1", "HW2"], False),
        (["HW1", None], ["HW1", None], True),
        (["HW1", "HW1"], ["HW1", "HW1"], True),
        (["HW1", "HW1"], ["HW1", "HW2"], False),
        (["HW1", None], ["HW1", "HW1"], True),
    ],
)
def test_transducer_array_pure_constructor_validates_device_identity(recorded_hwids, connected_hwids, valid):
    cfgs = [_example_module_user_config(str(i)) for i in range(len(connected_hwids))]
    for cfg, hwid in zip(cfgs, connected_hwids):
        if hwid is None:
            cfg.pop("hwid")
        else:
            cfg["hwid"] = hwid
    cfgs[0]["device"] = {"modules": [
        {"hwid": hwid} if hwid is not None else {} for hwid in recorded_hwids
    ]}
    if valid:
        array = TransducerArray.from_module_user_configs(cfgs)
        assert [m.attrs.get("hwid") for m in array.modules] == connected_hwids
    else:
        with pytest.raises(DeviceConfigMismatchError):
            TransducerArray.from_module_user_configs(cfgs)


@pytest.mark.parametrize("device", [None, {}])
def test_transducer_array_pure_constructor_accepts_no_device_metadata(device):
    cfg = _example_module_user_config()
    cfg["device"] = device
    assert len(TransducerArray.from_module_user_configs([cfg]).modules) == 1


@pytest.mark.parametrize("device", [{"id": "metadata_only"}, {"modules": []}])
def test_transducer_array_pure_constructor_rejects_device_without_modules(device):
    cfg = _example_module_user_config()
    cfg["device"] = device
    with pytest.raises(DeviceConfigMismatchError, match="lists 0 module"):
        TransducerArray.from_module_user_configs([cfg])


@pytest.mark.parametrize("as_list", [True, False])
def test_standoff_construction_merge_and_roundtrips(as_list):
    expected = _translation(8)
    value = expected.tolist() if as_list else expected.copy()
    transducer = Transducer.gen_matrix_array(nx=1, ny=1, standoff_transform=value)
    assert isinstance(transducer.standoff_transform, np.ndarray)
    np.testing.assert_array_equal(transducer.standoff_transform, expected)

    merged = Transducer.merge([transducer], merged_attrs={"standoff_transform": value})
    assert isinstance(merged.standoff_transform, np.ndarray)
    np.testing.assert_array_equal(merged.standoff_transform, expected)
    merged.standoff_transform[0, 3] = 99
    np.testing.assert_array_equal(value, expected)

    transducer.standoff_transform = value
    serialized = transducer.to_dict()
    assert isinstance(serialized["standoff_transform"], list)
    for restored in [Transducer.from_dict(serialized), Transducer.from_json(transducer.to_json())]:
        assert isinstance(restored.standoff_transform, np.ndarray)
        np.testing.assert_array_equal(restored.standoff_transform, expected)

    array = TransducerArray.from_module_user_configs([_example_module_user_config()])
    array.attrs["standoff_transform"] = value
    for restored in [array, TransducerArray.from_dict(json.loads(array.to_json()))]:
        flattened = restored.to_transducer()
        assert isinstance(flattened.standoff_transform, np.ndarray)
        np.testing.assert_array_equal(flattened.standoff_transform, expected)
        assert flattened.numelements() == 64


@pytest.mark.parametrize("standoff", [None, [], np.eye(3), np.ones((4, 3))])
def test_transducer_rejects_invalid_standoff_shape(standoff):
    with pytest.raises(ValueError, match="4x4"):
        Transducer(standoff_transform=standoff)
    with pytest.raises(ValueError, match="4x4"):
        Transducer.merge([Transducer()], merged_attrs={"standoff_transform": standoff})


def test_transducer_array_to_device_config_shape_and_independence():
    cfgs = [_example_module_user_config("HW2"), _example_module_user_config("HW1")]
    transforms = [_translation(2), _translation(1)]
    array = TransducerArray.from_module_user_configs(
        cfgs, arr_id="custom_array", arr_name="Custom Array", module_transforms=transforms,
    )
    array.attrs = {
        "standoff_transform": _translation(8),
        "weights": np.array([1.0, 2.0]),
        "metadata": {"labels": ["custom"]},
        "registration_surface_filename": "surface.obj",
    }
    original = copy.deepcopy(array)

    device = array.to_device_config()

    assert set(device) == {"id", "name", "modules", "attrs"}
    assert (device["id"], device["name"]) == ("custom_array", "Custom Array")
    assert device["modules"] == [
        {"hwid": "HW2", "transform": transforms[0].tolist()},
        {"hwid": "HW1", "transform": transforms[1].tolist()},
    ]
    assert device["attrs"]["standoff_transform"] == _translation(8).tolist()
    assert device["attrs"]["weights"] == [1.0, 2.0]
    assert device["attrs"]["registration_surface_filename"] == "surface.obj"
    assert json.loads(json.dumps(device)) == device
    assert dataclasses_are_equal(array, original)

    reconstructed_cfgs = copy.deepcopy(cfgs)
    reconstructed_cfgs[0]["device"] = device
    reconstructed = TransducerArray.from_module_user_configs(reconstructed_cfgs)
    assert reconstructed.to_device_config() == device
    original_reconstructed = copy.deepcopy(reconstructed)
    device["modules"][0]["transform"][0][3] = 99
    device["attrs"]["standoff_transform"][0][3] = 99
    device["attrs"]["metadata"]["labels"].append("changed")
    assert dataclasses_are_equal(array, original)
    assert dataclasses_are_equal(reconstructed, original_reconstructed)


def test_transducer_array_dict_serialization_does_not_alias_inputs():
    cfg = _example_module_user_config()
    cfg["module"]["attrs"] = {"calibration": {"values": [1]}}
    array = TransducerArray.from_module_user_configs([cfg])
    array.attrs = {"standoff_transform": _translation(8), "metadata": {"labels": ["custom"]}}
    original = copy.deepcopy(array)
    serialized = array.to_dict()
    assert dataclasses_are_equal(array, original)
    assert isinstance(serialized["attrs"]["standoff_transform"], list)
    serialized["attrs"]["impulse_response"] = [1, 2]
    serialized["attrs"]["impulse_dt"] = 1e-6
    original_serialized = copy.deepcopy(serialized)

    restored = TransducerArray.from_dict(serialized)

    assert serialized == original_serialized
    assert "impulse_response" not in restored.attrs
    assert "impulse_dt" not in restored.attrs
    assert isinstance(restored.attrs["standoff_transform"], np.ndarray)
    restored.attrs["metadata"]["labels"].append("changed")
    restored.modules[0].attrs["calibration"]["values"].append(2)
    restored.modules[0].module_invert[0] = True
    assert serialized == original_serialized
    serialized["attrs"]["standoff_transform"][0][3] = 99
    serialized["attrs"]["metadata"]["labels"].append("changed")
    serialized["modules"][0]["attrs"]["calibration"]["values"].append(2)
    serialized["modules"][0]["module_invert"][0] = True
    assert dataclasses_are_equal(array, original)


def _physical_module_configs(units):
    mm_per_unit = {"mm": 1, "cm": 10, "m": 1000}
    configs = [_example_module_user_config(f"HW{i}") for i in range(len(units))]
    for cfg, unit in zip(configs, units):
        cfg["module"].update(
            nx=2, ny=1, pitch=4 / mm_per_unit[unit], kerf=0.2 / mm_per_unit[unit], units=unit,
        )
    return configs


@pytest.mark.parametrize("template_units", [("mm", "mm"), ("cm", "cm"), ("mm", "cm")])
@pytest.mark.parametrize("module_units", [("mm", "mm"), ("cm", "cm"), ("cm", "mm")])
@pytest.mark.parametrize("as_lists", [False, True])
def test_template_geometry_preserves_physical_units(template_units, module_units, as_lists):
    template = TransducerArray.from_module_user_configs(_physical_module_configs(template_units))
    standoff = np.array([[1, 0, 0, 2], [0, 0, -1, 4], [0, 1, 0, 8], [0, 0, 0, 1]], dtype=float)
    array_standoff = standoff.copy()
    array_standoff[2, 3] = 18
    array_standoff[:3, 3] /= 1 if template_units[0] == "mm" else 10
    template.attrs["standoff_transform"] = array_standoff
    for i, module in enumerate(template.modules):
        mm_per_template_unit = 1 if module.units == "mm" else 10
        module.transform = np.array(
            [[0, -1, 0, 10 * (-1) ** i], [1, 0, 0, 4], [0, 0, 1, 6], [0, 0, 0, 1]], dtype=float,
        )
        module.transform[:3, 3] /= mm_per_template_unit
        module.standoff_transform = standoff.copy()
        module.standoff_transform[:3, 3] /= mm_per_template_unit
    reference = copy.deepcopy(template)
    if as_lists:
        template.attrs["standoff_transform"] = array_standoff.tolist()
        for module in template.modules:
            module.transform = module.transform.tolist()
            module.standoff_transform = module.standoff_transform.tolist()
    original_template = copy.deepcopy(template)
    configs = _physical_module_configs(module_units)
    original_configs = copy.deepcopy(configs)

    array = TransducerArray.from_module_user_configs(configs, template=template)

    for module, expected in zip(array.modules, reference.modules):
        np.testing.assert_allclose(module.bake().get_positions(units="mm"), expected.bake().get_positions(units="mm"))
        np.testing.assert_allclose(module.get_standoff_transform_in_units("mm"), expected.get_standoff_transform_in_units("mm"))
        np.testing.assert_array_equal(module.transform[:3, :3], expected.transform[:3, :3])
        assert module.frequency == expected.frequency
        assert module.sensitivity == expected.sensitivity
    assert [module.units for module in array.modules] == list(module_units)
    flattened = array.to_transducer()
    expected_flattened = reference.to_transducer()
    np.testing.assert_allclose(flattened.get_positions(units="mm"), expected_flattened.get_positions(units="mm"))
    np.testing.assert_allclose(flattened.get_standoff_transform_in_units("mm"), expected_flattened.get_standoff_transform_in_units("mm"))
    replay_configs = copy.deepcopy(configs)
    replay_configs[0]["device"] = array.to_device_config()
    replay = TransducerArray.from_module_user_configs(replay_configs, template=template)
    np.testing.assert_allclose(replay.to_transducer().get_positions(units="mm"), flattened.get_positions(units="mm"))
    np.testing.assert_allclose(replay.attrs["standoff_transform"], array.attrs["standoff_transform"])
    assert configs == original_configs
    assert dataclasses_are_equal(template, original_template)


@pytest.mark.parametrize("standoff_override", [None, _translation(3).tolist()])
def test_template_unit_conversion_preserves_device_and_explicit_overrides(standoff_override):
    template = TransducerArray.from_module_user_configs(_physical_module_configs(["mm"]))
    template.modules[0].transform = _translation(10)
    template.modules[0].standoff_transform = _translation(8)
    template.attrs["standoff_transform"] = _translation(8)
    configs = _physical_module_configs(["cm"])
    configs[0]["device"] = {
        "modules": [{"hwid": "HW0", "transform": _translation(2).tolist()}],
        "attrs": {"standoff_transform": standoff_override},
    }
    original_configs = copy.deepcopy(configs)
    array = TransducerArray.from_module_user_configs(configs, template=template)
    explicit_transform = _translation(4)
    explicit = TransducerArray.from_module_user_configs(configs, template=template, module_transforms=[explicit_transform])
    np.testing.assert_array_equal(array.modules[0].transform, _translation(2))
    np.testing.assert_array_equal(explicit.modules[0].transform, explicit_transform)
    for result in (array, explicit):
        np.testing.assert_array_equal(result.modules[0].standoff_transform, _translation(0.8))
        if standoff_override is None:
            assert result.attrs["standoff_transform"] is None
        else:
            np.testing.assert_array_equal(result.attrs["standoff_transform"], standoff_override)
    assert configs == original_configs


def test_translated_array_standoff_requires_template_units():
    template = TransducerArray(attrs={"standoff_transform": _translation(8)})
    configs = _physical_module_configs(["cm"])
    with pytest.raises(ValueError, match="standoff.*template.*module"):
        TransducerArray.from_module_user_configs(configs, template=template)
    template.attrs["standoff_transform"] = np.eye(4)
    array = TransducerArray.from_module_user_configs(configs, template=template)
    np.testing.assert_array_equal(array.attrs["standoff_transform"], np.eye(4))


@pytest.mark.parametrize("frequency", [155, 400])
def test_embedded_template_translations_are_in_millimeters(frequency):
    template = _build_meshless_default_template(f"openlifu_2x{frequency}")
    assert [module.units for module in template.modules] == ["mm", "mm"]
    array = TransducerArray.from_module_user_configs(_physical_module_configs(["cm", "cm"]), template=template)
    for module, template_module in zip(array.modules, template.modules):
        np.testing.assert_allclose(module.transform[:3, 3] * 10, template_module.transform[:3, 3])
        np.testing.assert_array_equal(module.transform[:3, :3], template_module.transform[:3, :3])
    assert array.modules[0].transform[0, 3] == pytest.approx(2.584571998794554)
    assert array.to_transducer().get_standoff_transform_in_units("mm")[2, 3] == pytest.approx(-8)


def test_template_geometry_rejects_incompatible_units():
    template = TransducerArray.from_module_user_configs(_physical_module_configs(["mm"]))
    template.modules[0].units = "s"
    with pytest.raises(ValueError, match="Unit type mismatch"):
        TransducerArray.from_module_user_configs(_physical_module_configs(["cm"]), template=template)
