from __future__ import annotations

import copy
import json
import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from helpers import dataclasses_are_equal

from openlifu.db import Database
from openlifu.xdc import (
    DeviceConfigMismatchError,
    Transducer,
    TransducerArray,
    arrays_structurally_equal,
)


def _module_user_config(hwid: str | None, freq: int = 400) -> dict:
    return {
        "hwid": hwid,
        "freq": freq,
        "module": {
            "nx": 2,
            "ny": 2,
            "pitch": 1.0,
            "kerf": 0.0,
            "units": "mm",
            "frequency": freq * 1000.0,
        },
    }


class _FakeTxDevice:
    def __init__(self, configs):
        self.configs = configs
        self.count_calls = 0
        self.read_calls = []

    def get_tx_module_count(self):
        self.count_calls += 1
        return len(self.configs)

    def read_config(self, *, module):
        self.read_calls.append(module)
        config = self.configs[module]
        if isinstance(config, Exception):
            raise config
        if config is None:
            return None
        payload = config if isinstance(config, str) else json.dumps(config)
        return SimpleNamespace(get_json_str=lambda: payload)


class _FakeInterface:
    def __init__(self, configs):
        self.txdevice = _FakeTxDevice(configs)
        self.close = Mock()


class _FakeDB:
    def __init__(self, templates=None, stored=None):
        self.templates = templates or {}
        self.stored = stored or {}
        self.load_calls = []

    def get_transducer_ids(self):
        return list(self.stored)

    def load_transducer(self, transducer_id, convert_array=True):
        self.load_calls.append((transducer_id, convert_array))
        value = self.templates.get(transducer_id, self.stored.get(transducer_id))
        if isinstance(value, Exception):
            raise value
        return value


@pytest.mark.parametrize(
    ("expected", "connected", "valid"),
    [
        (["AAA", "BBB"], ["AAA", "BBB"], True),
        (["BBB", "AAA"], ["AAA", "BBB"], True),
        (["AAA", "BBB"], ["AAA"], False),
        (["AAA", "BBB"], ["AAA", "ZZZ"], False),
        (["AAA", "BBB"], ["AAA", None], False),
        ([None, None], ["AAA", "BBB"], True),
        (["AAA", None], ["AAA", "BBB"], False),
        (["AAA", None], ["AAA", None], True),
        (["AAA", "AAA"], ["AAA", "AAA"], True),
        (["AAA", "AAA"], ["AAA", "BBB"], False),
        (["AAA", None], ["AAA", "AAA"], True),
        ([], ["AAA"], False),
    ],
)
def test_get_connected_validates_device_identity_before_database_lookup(expected, connected, valid):
    configs = [_module_user_config(hwid) for hwid in connected]
    configs[0]["device"] = {
        "id": "device",
        "modules": [{"hwid": hwid} if hwid else {} for hwid in expected],
    }
    interface = _FakeInterface(configs)
    db = _FakeDB()
    if valid:
        array = TransducerArray.get_connected(interface=interface, db=db)
        assert [module.attrs.get("hwid") for module in array.modules] == connected
    else:
        with pytest.raises(DeviceConfigMismatchError, match="lists .* module|HWIDs do not match"):
            TransducerArray.get_connected(interface=interface, db=db)
        assert db.load_calls == []
    assert interface.txdevice.read_calls == list(range(len(configs)))
    interface.close.assert_not_called()


@pytest.mark.parametrize("device", [None, {}])
def test_get_connected_accepts_absent_or_empty_device(device):
    config = _module_user_config("AAA")
    if device is not None:
        config["device"] = device
    array = TransducerArray.get_connected(interface=_FakeInterface([config]))
    assert array.id == "openlifu_1x400"


def test_get_connected_rejects_metadata_only_device():
    config = _module_user_config("AAA")
    config["device"] = {"id": "metadata-only", "name": "Metadata only"}
    with pytest.raises(DeviceConfigMismatchError, match="lists 0 module"):
        TransducerArray.get_connected(interface=_FakeInterface([config]))


@pytest.mark.parametrize(
    ("count", "freq", "template_id"),
    [(1, 155, "openlifu_1x155"), (1, 400, "openlifu_1x400"),
     (2, 155, "openlifu_2x155"), (2, 400, "openlifu_2x400")],
)
def test_get_connected_infers_embedded_template(count, freq, template_id):
    interface = _FakeInterface([_module_user_config(str(i), freq) for i in range(count)])
    array = TransducerArray.get_connected(interface=interface)
    assert array.id == template_id
    assert len(array.modules) == count
    assert array.registration_surface_filename is None
    assert array.transducer_body_filename is None
    assert isinstance(array.attrs["standoff_transform"], np.ndarray)
    assert all(module.frequency == freq * 1000 for module in array.modules)
    if count == 1:
        np.testing.assert_array_equal(array.modules[0].transform, np.eye(4))
    else:
        np.testing.assert_allclose(array.modules[0].transform[0, 3], 25.84571998794554)
        np.testing.assert_allclose(array.modules[1].transform[0, 3], -25.84571998794554)
    assert interface.txdevice.count_calls == 1
    assert interface.txdevice.read_calls == list(range(count))
    interface.close.assert_not_called()


@pytest.mark.parametrize("use_default_template", [True, False])
def test_get_connected_prefers_recorded_database_template(use_default_template):
    config = _module_user_config("AAA", 400)
    config["device"] = {
        "id": "device",
        "name": "Device",
        "template": "openlifu_1x155",
        "modules": [{"hwid": "AAA"}],
    }
    template = TransducerArray.from_module_user_configs([_module_user_config("template", 155)])
    template.modules[0].transform[0, 3] = 17.0
    template.modules[0].registration_surface_filename = "module-surface.obj"
    template.transducer_body_filename = "array-body.obj"
    original = copy.deepcopy(template)
    db = _FakeDB(templates={"openlifu_1x155": template})
    array = TransducerArray.get_connected(
        interface=_FakeInterface([config]), db=db, use_default_template=use_default_template,
    )
    assert (array.id, array.name) == ("device", "Device")
    assert array.modules[0].frequency == 400000.0
    assert array.modules[0].registration_surface_filename == "module-surface.obj"
    assert array.transducer_body_filename == "array-body.obj"
    assert array.modules[0].transform[0, 3] == 17.0
    assert db.load_calls == [("openlifu_1x155", False)]
    assert dataclasses_are_equal(template, original)


@pytest.mark.parametrize("use_default_template", [True, False])
@pytest.mark.parametrize("loaded", [None, FileNotFoundError("missing template"), Transducer()])
def test_get_connected_database_template_unavailable(loaded, use_default_template):
    db = _FakeDB(templates={"openlifu_1x400": loaded})
    array = TransducerArray.get_connected(
        interface=_FakeInterface([_module_user_config("AAA")]), db=db,
        use_default_template=use_default_template,
    )
    assert array.id == ("openlifu_1x400" if use_default_template else "transducer_array")
    assert ("standoff_transform" in array.attrs) is use_default_template
    assert db.load_calls == [("openlifu_1x400", False)]


@pytest.mark.parametrize("use_default_template", [True, False])
def test_get_connected_unknown_recorded_template_does_not_infer_replacement(use_default_template):
    config = _module_user_config("AAA")
    config["device"] = {
        "id": "device", "template": "custom-template", "modules": [{"hwid": "AAA"}],
    }
    db = _FakeDB()
    array = TransducerArray.get_connected(
        interface=_FakeInterface([config]), db=db, use_default_template=use_default_template,
    )
    assert db.load_calls == [("custom-template", False)]
    assert array.id == "device"
    assert "standoff_transform" not in array.attrs
    np.testing.assert_array_equal(array.modules[0].transform, np.eye(4))


@pytest.mark.parametrize("freq", [None, 250])
def test_get_connected_unknown_frequency_constructs_without_template(freq):
    config = _module_user_config("AAA")
    config["freq"] = freq
    db = _FakeDB()
    array = TransducerArray.get_connected(interface=_FakeInterface([config]), db=db)
    assert array.id == "transducer_array"
    assert db.load_calls == []
    np.testing.assert_array_equal(array.modules[0].transform, np.eye(4))


def test_get_connected_forwards_explicit_overrides():
    configs = [_module_user_config("AAA"), _module_user_config("BBB")]
    transforms = [np.eye(4), np.eye(4)]
    transforms[0][0, 3] = 1.0
    transforms[1][0, 3] = 2.0
    array = TransducerArray.get_connected(
        interface=_FakeInterface(configs), arr_id="custom", arr_name="Custom array",
        module_transforms=transforms,
    )
    assert (array.id, array.name) == ("custom", "Custom array")
    for module, transform in zip(array.modules, transforms):
        np.testing.assert_array_equal(module.transform, transform)


@pytest.mark.parametrize("recorded_template", [None, "openlifu_2x400"])
def test_get_connected_checks_frequencies_before_template_lookup(recorded_template):
    configs = [_module_user_config("AAA", 400), _module_user_config("BBB", 155)]
    if recorded_template:
        configs[0]["device"] = {
            "template": recorded_template, "modules": [{"hwid": "AAA"}, {"hwid": "BBB"}],
        }
    db = _FakeDB()
    with pytest.raises(ValueError, match="mismatched frequencies"):
        TransducerArray.get_connected(interface=_FakeInterface(configs), db=db)
    assert db.load_calls == []


@pytest.mark.parametrize("owned", [True, False])
@pytest.mark.parametrize("outcome", ["success", "empty", "count-error", "none", "read-error", "json", "frequency", "identity", "module"])
def test_get_connected_closes_only_owned_interface(monkeypatch, owned, outcome):
    configs = [_module_user_config("AAA")]
    error = RuntimeError
    message = ""
    if outcome == "empty":
        configs = []
        message = "No TX modules"
    elif outcome == "none":
        configs = [None]
        message = "module 0"
    elif outcome == "read-error":
        configs = [OSError("USB read failed")]
        error, message = OSError, "USB read failed"
    elif outcome == "json":
        configs = ["{malformed"]
        error, message = json.JSONDecodeError, "Expecting"
    elif outcome == "frequency":
        configs.append(_module_user_config("BBB", 155))
        error, message = ValueError, "mismatched frequencies"
    elif outcome == "identity":
        configs[0]["device"] = {"modules": [{"hwid": "ZZZ"}]}
        error, message = DeviceConfigMismatchError, "HWIDs do not match"
    elif outcome == "module":
        configs[0]["module"] = {}
        error, message = ValueError, "module"
    interface = _FakeInterface(configs)
    if outcome == "count-error":
        interface.txdevice.get_tx_module_count = Mock(side_effect=OSError("Module count failed"))
        error, message = OSError, "Module count failed"
    factory = Mock(return_value=interface)
    sdk_io = ModuleType("openlifu_sdk.io")
    sdk_io.LIFUInterface = factory
    monkeypatch.setitem(sys.modules, "openlifu_sdk.io", sdk_io)
    kwargs = {} if owned else {"interface": interface}
    if outcome == "success":
        array = TransducerArray.get_connected(**kwargs)
        assert array.modules[0].attrs["hwid"] == "AAA"
    else:
        with pytest.raises(error, match=message):
            TransducerArray.get_connected(**kwargs)
    if owned:
        factory.assert_called_once_with()
        interface.close.assert_called_once_with()
    else:
        factory.assert_not_called()
        interface.close.assert_not_called()


@pytest.mark.parametrize("import_failure", ["absent", "dependency", "internal"])
def test_optional_sdk_import_in_fresh_process(import_failure):
    script = textwrap.dedent("""
        import importlib.abc
        import json
        import sys
        from types import SimpleNamespace

        mode = sys.argv[1]
        attempts = []

        class BlockSDK(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "openlifu_sdk" or fullname.startswith("openlifu_sdk."):
                    attempts.append(fullname)
                    if mode == "absent":
                        raise ModuleNotFoundError("SDK is absent", name="openlifu_sdk")
                    if mode == "dependency":
                        raise ModuleNotFoundError("SDK dependency is absent", name="sdk_dependency")
                    raise ImportError("SDK internal failure")

        sys.meta_path.insert(0, BlockSDK())
        from openlifu.xdc import Transducer, TransducerArray

        config = json.loads(sys.argv[2])
        module = Transducer.from_module_user_config(config)
        pure = TransducerArray.from_module_user_configs([config])
        pure.to_device_config()
        interface = SimpleNamespace(txdevice=SimpleNamespace(
            get_tx_module_count=lambda: 1,
            read_config=lambda module: SimpleNamespace(get_json_str=lambda: json.dumps(config)),
        ))
        connected = TransducerArray.get_connected(interface=interface)
        assert module.numelements() == 4
        assert connected.modules[0].attrs["hwid"] == "AAA"
        assert attempts == [], attempts
        try:
            TransducerArray.get_connected()
        except ImportError as exc:
            if mode == "absent":
                assert "openlifu_sdk" in str(exc), str(exc)
                assert "interface" in str(exc), str(exc)
            elif mode == "dependency":
                assert isinstance(exc, ModuleNotFoundError), repr(exc)
                assert exc.name == "sdk_dependency", repr(exc)
                assert str(exc) == "SDK dependency is absent", str(exc)
            else:
                assert str(exc) == "SDK internal failure", str(exc)
        else:
            raise AssertionError("SDK import failure was swallowed")
        assert attempts
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, import_failure, json.dumps(_module_user_config("AAA"))],
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("different", [True, False])
def test_get_connected_database_comparison_warning(different):
    configs = [_module_user_config("AAA")]
    configs[0]["device"] = {
        "id": "device", "name": "Device", "template": "openlifu_1x400",
        "modules": [{"hwid": "AAA"}],
    }
    stored = TransducerArray.get_connected(interface=_FakeInterface(configs))
    if different:
        stored.name = "Other array"
    original = copy.deepcopy(stored)
    db = _FakeDB(stored={stored.id: stored})
    if different:
        with pytest.warns(UserWarning, match="differs from the version"):
            array = TransducerArray.get_connected(interface=_FakeInterface(configs), db=db)
    else:
        array = TransducerArray.get_connected(interface=_FakeInterface(configs), db=db)
    assert arrays_structurally_equal(array, stored) is not different
    assert dataclasses_are_equal(stored, original)


def test_arrays_structurally_equal_normalizes_without_mutating_inputs():
    array = TransducerArray.get_connected(interface=_FakeInterface([_module_user_config("AAA")]))
    array.registration_surface_filename = "surface.obj"
    array.transducer_body_filename = "body.obj"
    array.modules[0].registration_surface_filename = "module-surface.obj"
    array.modules[0].transducer_body_filename = "module-body.obj"
    array.attrs["samples"] = np.array([1.0, 2.0])
    other = copy.deepcopy(array)
    other.registration_surface_filename = "/database/device/surface.obj"
    other.transducer_body_filename = "/database/device/body.obj"
    other.modules[0].registration_surface_filename = "/database/module/module-surface.obj"
    other.modules[0].transducer_body_filename = "/database/module/module-body.obj"
    other.attrs["samples"] = [1.0000001, 2.0000001]
    other.attrs["standoff_transform"] = other.attrs["standoff_transform"].tolist()
    other.modules[0].transform[0, 3] += 0.0000001
    for attrs in (other.attrs, other.modules[0].attrs):
        attrs["impulse_response"] = [123.0]
        attrs["impulse_dt"] = 0.123
    original_array, original_other = copy.deepcopy((array, other))
    assert arrays_structurally_equal(array, other)
    assert arrays_structurally_equal(other, array)
    assert dataclasses_are_equal(array, original_array)
    assert dataclasses_are_equal(other, original_other)


@pytest.mark.parametrize("difference", ["name", "hwid", "geometry", "transform", "mesh", "attrs"])
def test_arrays_structurally_equal_detects_meaningful_differences(difference):
    array = TransducerArray.from_module_user_configs([_module_user_config("AAA")])
    other = copy.deepcopy(array)
    if difference == "name":
        other.name = "Different name"
    elif difference == "hwid":
        other.modules[0].attrs["hwid"] = "BBB"
    elif difference == "geometry":
        other.modules[0].elements[0].position[0] += 0.001
    elif difference == "transform":
        other.modules[0].transform[0, 3] += 0.000001
    elif difference == "mesh":
        other.registration_surface_filename = "different.obj"
    else:
        other.attrs["custom"] = True
    assert not arrays_structurally_equal(array, other)


def test_to_device_config_has_independent_json_compatible_data():
    configs = [_module_user_config("BBB"), _module_user_config("AAA")]
    array = TransducerArray.get_connected(interface=_FakeInterface(configs), arr_id="device", arr_name="Device")
    array.registration_surface_filename = "surface.obj"
    array.transducer_body_filename = "body.obj"
    array.attrs["metadata"] = {"labels": ["original"]}
    array.attrs["samples"] = np.array([1.0, 2.0])
    original = copy.deepcopy(array)
    serialized = array.to_device_config()
    assert set(serialized) == {"id", "name", "modules", "attrs"}
    assert (serialized["id"], serialized["name"]) == ("device", "Device")
    assert [module["hwid"] for module in serialized["modules"]] == ["BBB", "AAA"]
    assert all(set(module) == {"hwid", "transform"} for module in serialized["modules"])
    assert serialized["attrs"]["samples"] == [1.0, 2.0]
    assert json.loads(json.dumps(serialized)) == serialized
    for entry, module in zip(serialized["modules"], array.modules):
        np.testing.assert_array_equal(entry["transform"], module.transform)
    np.testing.assert_array_equal(serialized["attrs"]["standoff_transform"], array.attrs["standoff_transform"])
    configs[0]["device"] = copy.deepcopy(serialized)
    rebuilt = TransducerArray.from_module_user_configs(configs)
    assert arrays_structurally_equal(array, rebuilt)
    serialized["attrs"]["metadata"]["labels"].append("changed")
    serialized["attrs"]["standoff_transform"][0][3] = 999.0
    serialized["modules"][0]["transform"][0][3] = 999.0
    assert dataclasses_are_equal(array, original)


def test_connected_array_saves_loads_and_flattens_with_temporary_mesh_files(tmp_path):
    db = Database.initialize_empty_database(tmp_path / "db")
    configs = [_module_user_config("AAA"), _module_user_config("BBB")]
    array = TransducerArray.get_connected(interface=_FakeInterface(configs), arr_id="device")
    mesh_text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    surface_path = tmp_path / "surface.obj"
    body_path = tmp_path / "body.obj"
    surface_path.write_text(mesh_text)
    body_path.write_text(mesh_text)
    db.write_transducer(array, surface_path, body_path)
    loaded = db.load_transducer(array.id, convert_array=False)
    assert isinstance(loaded, TransducerArray)
    assert arrays_structurally_equal(array, loaded)
    paths = db.get_transducer_absolute_filepaths(array.id)
    assert (tmp_path / paths["registration_surface_abspath"]).read_text() == mesh_text
    assert (tmp_path / paths["transducer_body_abspath"]).read_text() == mesh_text
    flattened = db.load_transducer(array.id)
    assert isinstance(flattened, Transducer)
    assert flattened.numelements() == 8
    assert flattened.registration_surface_filename == surface_path.name
    np.testing.assert_array_equal(flattened.standoff_transform, array.attrs["standoff_transform"])
    expected_positions = np.concatenate([module.bake().get_positions() for module in array.modules])
    np.testing.assert_allclose(flattened.get_positions(), expected_positions)
