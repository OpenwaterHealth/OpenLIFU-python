from __future__ import annotations

import copy
import json
import os
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from openlifu.util.dict_conversion import DictMixin
from openlifu.util.units import getunitconversion
from openlifu.xdc import Transducer, TransformedTransducer

# Mapping from (num_connected_modules, freq_khz) to the canonical
# template id consumed by :py:meth:`TransducerArray.get_connected`.
_DEFAULT_TEMPLATE_IDS: dict[tuple[int, int], str] = {
    (1, 155): "openlifu_1x155",
    (1, 400): "openlifu_1x400",
    (2, 155): "openlifu_2x155",
    (2, 400): "openlifu_2x400",
}

# Locally-embedded per-module transforms and array-level standoff for
# the canonical default templates. Used as a meshless fallback when no
# database is provided to :py:meth:`TransducerArray.get_connected`.
# Translations are in millimeters.
# The 2x155 entries currently mirror the openlifu_2x180_evt1 template
# as a stand-in until a dedicated 155 kHz template ships.
_DEFAULT_TEMPLATE_DATA: dict[str, dict] = {
    "openlifu_1x155": {
        "name": "OpenLIFU 1x 155kHz",
        "module_transforms": [np.eye(4, dtype=float)],
        "standoff_transform": np.eye(4, dtype=float),
    },
    "openlifu_1x400": {
        "name": "OpenLIFU 1x 400kHz",
        "module_transforms": [np.eye(4, dtype=float)],
        "standoff_transform": np.eye(4, dtype=float),
    },
    "openlifu_2x155": {
        "name": "OpenLIFU 2x 155kHz",
        "module_transforms": [
            np.array([
                [0.9697859993972769, 0.0, -0.2439571998794554, 25.84571998794554],
                [0.0, 1.0, 0.0, 0.0],
                [0.24395719987945538, 0.0, 0.9697859993972772, 3.20098197421292],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=float),
            np.array([
                [0.9697859993972769, 0.0, 0.2439571998794554, -25.84571998794554],
                [0.0, 1.0, 0.0, 0.0],
                [-0.24395719987945538, 0.0, 0.9697859993972772, 3.20098197421292],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=float),
        ],
        "standoff_transform": np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.997684, -0.0680153, 0.0],
            [0.0, 0.0680153, 0.997684, -8.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float),
    },
    "openlifu_2x400": {
        "name": "OpenLIFU 2x 400kHz",
        "module_transforms": [
            np.array([
                [0.9659258262890683, 0.0, -0.25881904510252074, 25.84571998794554],
                [0.0, 1.0, 0.0, 0.0],
                [0.25881904510252074, 0.0, 0.9659258262890683, 3.20098197421292],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=float),
            np.array([
                [0.9659258262890683, 0.0, 0.25881904510252074, -25.84571998794554],
                [0.0, 1.0, 0.0, 0.0],
                [-0.25881904510252074, 0.0, 0.9659258262890683, 3.20098197421292],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=float),
        ],
        "standoff_transform": np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -8.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float),
    },
}


class DeviceConfigMismatchError(ValueError):
    """Raised when stored device identity differs from the module configurations.

    Both :py:meth:`TransducerArray.get_connected` and
    :py:meth:`TransducerArray.from_module_user_configs` validate that block
    against the supplied module configurations before assembling an array.
    """


def _validate_device_config_against_connected(
    device_cfg: dict,
    user_configs: Sequence[dict],
) -> None:
    """Check a stored ``device`` block matches the connected modules.

    The recorded module count must match. When any expected HWIDs are
    recorded, compare their set with all reported HWIDs, ignoring order.
    Partially populated HWIDs can therefore fail this check. With no expected
    HWIDs, only the count is checked.

    Raises:
        DeviceConfigMismatchError: if the count or the HWID sets disagree.
    """
    expected_modules = list(device_cfg.get("modules") or [])
    if len(expected_modules) != len(user_configs):
        raise DeviceConfigMismatchError(
            f"Device config '{device_cfg.get('id')}' lists {len(expected_modules)} "
            f"module(s) but {len(user_configs)} module(s) are connected."
        )

    expected_hwids = {
        m["hwid"]
        for m in expected_modules
        if isinstance(m, dict) and m.get("hwid")
    }
    if not expected_hwids:
        # No HWIDs to compare; count match alone is acceptable.
        return
    connected_hwids = {
        c.get("hwid") for c in user_configs if c.get("hwid")
    }
    missing = expected_hwids - connected_hwids
    extra = connected_hwids - expected_hwids
    if missing or extra:
        raise DeviceConfigMismatchError(
            f"Device config '{device_cfg.get('id')}' HWIDs do not match connected "
            f"hardware. Missing from connected: {sorted(missing)!r}; "
            f"unexpected on connected: {sorted(extra)!r}."
        )


def _validate_template_mesh_units(
    template: TransducerArray | None,
    modules: Sequence[TransformedTransducer],
    attrs: dict,
) -> None:
    """Reject inherited mesh coordinates that would require rescaling."""
    if template is None:
        return
    mesh_fields = ("registration_surface_filename", "transducer_body_filename")
    for index, (source, destination) in enumerate(zip(template.modules, modules)):
        for mesh_field in mesh_fields:
            if getattr(source, mesh_field) and getunitconversion(source.units, destination.units) != 1:
                raise ValueError(
                    f"Mesh {mesh_field} for module {index} cannot be inherited across units "
                    f"({source.units} to {destination.units}); mesh rescaling is not supported."
                )
    for mesh_field in mesh_fields:
        if not template.attrs.get(mesh_field) or not attrs.get(mesh_field):
            continue
        if not template.modules:
            raise ValueError(f"Mesh {mesh_field} units require a template module.")
        source_units = template.modules[0].units
        destination_units = modules[0].units
        if getunitconversion(source_units, destination_units) != 1:
            raise ValueError(
                f"Mesh {mesh_field} for the array cannot be inherited across units "
                f"({source_units} to {destination_units}); mesh rescaling is not supported."
            )


def _find_module_matching(candidates: list[list[int]], fixed: dict[int, int]) -> list[int] | None:
    """Find a one-to-one reported-to-recorded assignment, retaining fixed pairs."""
    if len(set(fixed.values())) != len(fixed):
        return None
    if any(recorded not in candidates[reported] for reported, recorded in fixed.items()):
        return None
    owners = {recorded: reported for reported, recorded in fixed.items()}

    def assign(reported, visited):
        available = candidates[reported]
        for recorded in available:
            if recorded not in visited and recorded not in owners:
                owners[recorded] = reported
                return True
        for recorded in available:
            if recorded in visited:
                continue
            visited.add(recorded)
            owner = owners[recorded]
            if owner not in fixed and assign(owner, visited):
                owners[recorded] = reported
                return True
        return False

    for reported in range(len(candidates)):
        if reported not in fixed and not assign(reported, set()):
            return None
    matches = [0] * len(candidates)
    for recorded, reported in owners.items():
        matches[reported] = recorded
    return matches


def _associate_device_modules(device_modules: list[dict], user_configs: Sequence[dict]) -> tuple[list[int], list[list[int]]]:
    """Return an assignment and every feasible origin of each recorded module.

    Known IDs must agree; absent IDs permit positional fallback. IDs unique in
    both lists are reserved before assigning other entries. Candidate order
    prefers the same position, but no recorded entry may be assigned twice.
    """
    if any(not isinstance(module, dict) for module in device_modules):
        raise DeviceConfigMismatchError("Cannot associate device modules: entries must be dictionaries.")
    reported_ids = [cfg.get("hwid") for cfg in user_configs]
    recorded_ids = [module.get("hwid") for module in device_modules]
    reported_counts, recorded_counts = Counter(reported_ids), Counter(recorded_ids)
    fixed = {
        i: recorded_ids.index(hwid)
        for i, hwid in enumerate(reported_ids)
        if hwid and reported_counts[hwid] == 1 and recorded_counts[hwid] == 1
    }
    candidates = [
        sorted(
            [j for j, recorded in enumerate(recorded_ids) if not reported or not recorded or reported == recorded],
            key=lambda j, i=i: (j != i, j),
        )
        for i, reported in enumerate(reported_ids)
    ]
    matches = _find_module_matching(candidates, fixed)
    if matches is None:
        raise DeviceConfigMismatchError("Cannot associate device modules without reusing entries or mismatching hardware IDs.")

    origins: list[list[int]] = [[] for _ in device_modules]
    for recorded in range(len(device_modules)):
        for reported, possible in enumerate(candidates):
            if recorded not in possible or (reported in fixed and fixed[reported] != recorded):
                continue
            if _find_module_matching(candidates, {**fixed, reported: recorded}) is not None:
                origins[recorded].append(reported)
    return matches, origins


def _recorded_module_units(recorded: int, origins: list[list[int]], modules: list[TransformedTransducer]) -> str:
    """Resolve units only when all feasible origins use the same scale."""
    units = modules[origins[recorded][0]].units
    if any(getunitconversion(modules[i].units, units) != 1 for i in origins[recorded]):
        raise DeviceConfigMismatchError(f"Ambiguous units for recorded device module {recorded}.")
    return units


def _build_meshless_default_template(template_id: str) -> TransducerArray:
    """Build a meshless template :class:`TransducerArray` from embedded transforms."""
    spec = _DEFAULT_TEMPLATE_DATA[template_id]
    modules: list[TransformedTransducer] = []
    for tform in spec["module_transforms"]:
        t = Transducer(id=template_id, elements=[], units="mm")
        modules.append(TransformedTransducer.from_transducer(t, transform=np.array(tform, dtype=float)))
    attrs = {"standoff_transform": np.array(spec["standoff_transform"], dtype=float)}
    return TransducerArray(id=template_id, name=spec["name"], modules=modules, attrs=attrs)


def _canonicalize_array_for_compare(arr: TransducerArray) -> dict:
    """Produce a structure suitable for equality-comparing two :class:`TransducerArray`.

    Normalizations applied:

    * NumPy arrays are converted to nested lists and rounded so trivial
      floating-point noise does not trigger spurious mismatches.
    * Mesh filename fields (``registration_surface_filename``,
      ``transducer_body_filename``) are reduced to their basename so absolute
      vs database-relative paths are treated as equivalent.
    * Fields that legitimately vary between a reconstructed array and a
      database-stored one (e.g. ``impulse_response`` / ``impulse_dt`` from
      calibration) are dropped.

    Used by :py:meth:`TransducerArray.get_connected` to warn when the array
    assembled from connected hardware disagrees with the same-id array in
    the supplied database.
    """
    def _norm(obj):
        if isinstance(obj, np.ndarray):
            return _norm(obj.tolist())
        if isinstance(obj, list | tuple):
            return [_norm(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _norm(v) for k, v in obj.items()}
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    raw = _norm(arr.to_dict())
    # Strip per-module fields that do not need to round-trip identically.
    for m in raw.get("modules", []):
        for k in ("registration_surface_filename", "transducer_body_filename"):
            v = m.get(k)
            if isinstance(v, str) and v:
                m[k] = os.path.basename(v)
        attrs = m.get("attrs") or {}
        attrs.pop("impulse_response", None)
        attrs.pop("impulse_dt", None)
    # Strip array-level mesh paths likewise.
    arr_attrs = raw.get("attrs") or {}
    for k in ("registration_surface_filename", "transducer_body_filename"):
        v = arr_attrs.get(k)
        if isinstance(v, str) and v:
            arr_attrs[k] = os.path.basename(v)
    arr_attrs.pop("impulse_response", None)
    arr_attrs.pop("impulse_dt", None)
    return raw


def arrays_structurally_equal(a: TransducerArray, b: TransducerArray) -> bool:
    """Return ``True`` if two arrays are equal after :func:`_canonicalize_array_for_compare`."""
    return _canonicalize_array_for_compare(a) == _canonicalize_array_for_compare(b)


def get_angle_from_gap(width, gap, roc):
    a = roc
    b = width/2
    c = gap/2
    mag = np.sqrt(a**2 + b**2)
    A = a/mag
    B = b/mag
    dth = np.arcsin(c/mag) + np.arcsin(B)
    return dth if A >= 0 else -dth

def get_roc_from_angle(width, gap, dth):
    return (0.5*gap + (0.5 * width * np.cos(dth))) / np.sin(dth)

def get_gap_from_angle(width, dth, roc):
    a = roc
    b = width/2
    mag = np.sqrt(a**2 + b**2)
    A = a/mag
    B = b/mag
    gap = 2*mag*np.sin(dth - np.arcsin(B))
    return gap if A >= 0 else -gap

@dataclass
class TransducerArray(DictMixin):
    id: str = "transducer_array"
    name: str = "Transducer Array"
    modules: list[TransformedTransducer] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)

    def to_transducer(self, offset_pins=True, offset_indices=True):
        t = Transducer.merge([t.bake() for t in self.modules], offset_pins=offset_pins, offset_indices=offset_indices, merged_attrs=self.attrs)
        t.name = self.name
        t.id = self.id
        return t

    @staticmethod
    def from_dict(data: dict):
        d = copy.deepcopy(data)
        if "type" in d:
            d.pop("type")
        d["modules"] = [TransformedTransducer.from_dict(t) for t in d["modules"]]
        if "attrs" in d:
            if "standoff_transform" in d["attrs"] and d["attrs"]["standoff_transform"] is not None:
                d["attrs"]["standoff_transform"] = np.array(d["attrs"]["standoff_transform"])
            d["attrs"].pop("impulse_response", None)
            d["attrs"].pop("impulse_dt", None)
        return TransducerArray(**d)


    def to_dict(self):
        d = {"type": "TransducerArray"}
        d.update(self.__dict__)
        d["modules"] = [t.to_dict() for t in self.modules]
        d = copy.deepcopy(d)
        for k, v in d["attrs"].items():
            if isinstance(v, np.ndarray):
                d["attrs"][k] = v.tolist()
        return d

    def to_json(self, compact:bool=False) -> str:
        """Serialize a TransducerArray to a json string

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete TransducerArray object.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'))
        else:
            return json.dumps(self.to_dict(), indent=4)

    def to_file(self, file_path: str, compact: bool = False) -> None:
        """Serialize a TransducerArray to a json file

        Args:
            file_path: The path to the file where the json string will be written.
            compact: if enabled then the string is compact (not pretty). Disable for pretty.
        """
        json_string = self.to_json(compact=compact)
        with open(file_path, 'w') as f:
            f.write(json_string)

    @staticmethod
    def get_concave_cylinder(trans, rows=1, cols=1, width=40, gap=None, dth=None, roc=None, units="mm", id="transducer_array", name="Transducer Array", attrs: dict={}):

        modules = []
        if isinstance(trans, Transducer):
            trans_arr = np.array([[trans]*cols for _ in range(rows)])
        else:
            trans_arr = np.array(trans).reshape(rows, cols)
        scl = getunitconversion(units, trans_arr[0,0].units)
        if gap is None:
            if dth is not None and roc is not None:
                gap = get_gap_from_angle(width, dth, roc)
            else:
                gap = 0
        elif dth is not None and roc is not None:
            raise ValueError("Invalid combination of parameters: cannot specify all of gap, dth, and roc.")

        if dth is None:
            if roc is not None:
                dth = get_angle_from_gap(width, gap, roc)
            else:
                dth = 0
                roc = np.inf

        if roc is None:
            if np.isclose(dth, 0.0):
                roc = np.inf
            else:
                roc = get_roc_from_angle(width, gap, dth)

        if dth == 0:
            for i in range(rows):
                y = (width+gap)*(i-(rows-1)/2)*scl
                for j in range(cols):
                    dx = (width+gap)*(j-(cols-1)/2)*scl
                    M = np.array([[1,0,0,dx], [0,1,0,y], [0,0,1,0], [0,0,0,1]])
                    trans_new = TransformedTransducer.from_transducer(trans_arr[i,j], transform=np.linalg.inv(M))
                    modules.append(trans_new)
        else:
            for i in range(rows):
                y = (width+gap)*(i-(rows-1)/2)*scl
                for j in range(cols):
                    th = dth*2*(j-(cols-1)/2)
                    x = roc*np.sin(th)*scl
                    z = roc*(1-np.cos(th))*scl
                    M = np.array([[np.cos(th),0,-np.sin(th),x],
                                [0,1,0,y],
                                [np.sin(th),0,np.cos(th),z],
                                [0,0,0,1]])
                    trans_new = TransformedTransducer.from_transducer(trans_arr[i,j], transform=np.linalg.inv(M))
                    modules.append(trans_new)
        return TransducerArray(modules=modules, id=id, name=name, attrs=attrs)

    @staticmethod
    def from_file(filename: str) -> TransducerArray:
        with open(filename) as f:
            data = json.load(f)
        return TransducerArray.from_dict(data)

    @classmethod
    def from_module_user_configs(
        cls,
        user_configs: Sequence[dict],
        template: TransducerArray | None = None,
        module_transforms: Sequence[np.ndarray] | None = None,
        arr_id: str | None = None,
        arr_name: str | None = None,
    ) -> TransducerArray:
        """Construct a :class:`TransducerArray` from one or more module ``user_config`` dicts.

        Each ``user_config`` describes a single physical module as reported by
        the SDK (``hwid``, ``module`` sub-dict suitable for
        :py:meth:`Transducer.gen_matrix_array`, optional ``device`` sub-dict on
        the lead module). User configs cannot carry mesh data, so a
        ``template`` :class:`TransducerArray` is normally supplied to inject
        per-module mesh filenames / standoff transforms / placement transforms
        and array-level metadata (id, name, attrs).

        Sources of array-level metadata, lowest priority first:

        1. ``template``: provides ``id``, ``name``, ``attrs``, and per-module
           ``transform``, ``standoff_transform``, ``registration_surface_filename``,
           ``transducer_body_filename``. Modules are matched to ``user_configs``
           positionally.
        2. ``user_configs[0]["device"]`` (if present): overrides ``id``,
           ``name``, merges into ``attrs``, and supplies per-module transforms
           keyed by ``hwid`` when unique in both the configs and device
           entries. Remaining entries are matched one-to-one, preferring
           position when IDs agree or either ID is absent.
        3. ``module_transforms`` (if given): per-module 4x4 transforms that
           override everything else. Length must match ``user_configs``.
        4. ``arr_id`` / ``arr_name`` (if given): explicit array id/name that
           override the values picked up from the device config or template.

        The per-module ``Transducer`` is always rebuilt from the user_config's
        ``module`` field (this is the on-device truth for nx/ny/pitch/kerf/
        frequency/sensitivity/etc.); only metadata that cannot live in the
        user_config is taken from the template.

        A nonempty lead-module ``device`` block must record a matching module
        count. If any expected HWIDs are recorded, their set must match all
        reported HWIDs. A metadata-only block without module entries fails
        count validation; an absent or empty block is valid. Associations that
        require reusing an entry or pairing different known IDs are rejected.

        Inherited placement and standoff translations are converted from each
        template module's units to the corresponding configuration's units.
        Array-level standoff uses the first module's units in each array.
        A translated template array standoff requires a template module to
        establish its units. Device array standoff uses the recorded first
        module's units, inferred from its possible matches. Ambiguous stored
        translations with possible origins in different units are rejected.
        Each physical module must retain its units since the device block was
        recorded; explicit transforms use the destination module units.

        Mesh references cannot be inherited across different unit scales
        because their coordinates are not rescaled. Renaming a template array
        mesh through device attributes does not bypass this check. An explicit
        ``None`` array standoff means identity and overrides any template standoff.

        Args:
            user_configs: ordered list of user_config dicts. Order corresponds
                to module index as reported by the device.
            template: optional template array; see above for what it supplies.
            module_transforms: optional list of explicit 4x4 transforms,
                one per user_config, that override template/device transforms.
            arr_id: optional explicit array id. Highest-priority source for the
                resulting ``TransducerArray.id`` (overrides device/template/default).
            arr_name: optional explicit array name. Highest-priority source for
                the resulting ``TransducerArray.name``.

        Returns:
            A :class:`TransducerArray` whose ``modules`` are
            :class:`TransformedTransducer` instances built from the
            user_configs.
        """
        if not user_configs:
            raise ValueError("user_configs must contain at least one user_config dict")
        if module_transforms is not None and len(module_transforms) != len(user_configs):
            raise ValueError(
                f"module_transforms length ({len(module_transforms)}) does not match "
                f"user_configs length ({len(user_configs)})"
            )

        resolved_id: str = "transducer_array"
        resolved_name: str = "Transducer Array"
        arr_attrs: dict = {}
        if template is not None:
            resolved_id = template.id
            resolved_name = template.name
            arr_attrs = copy.deepcopy(template.attrs)

        device_cfg = user_configs[0].get("device") or None
        device_attrs = (device_cfg or {}).get("attrs") or {}
        device_modules_in_order: list = []
        device_module_indices: list[int] = []
        device_module_origins: list[list[int]] = []
        if device_cfg:
            _validate_device_config_against_connected(device_cfg, user_configs)
            resolved_id = device_cfg.get("id", resolved_id)
            resolved_name = device_cfg.get("name", resolved_name)
            for k, v in device_attrs.items():
                arr_attrs[k] = copy.deepcopy(v)
            device_modules_in_order = list(device_cfg.get("modules") or [])
            device_module_indices, device_module_origins = _associate_device_modules(device_modules_in_order, user_configs)

        if arr_id is not None:
            resolved_id = arr_id
        if arr_name is not None:
            resolved_name = arr_name

        template_modules: list = list(template.modules) if template is not None else []
        modules: list[TransformedTransducer] = []
        for i, cfg in enumerate(user_configs):
            t = Transducer.from_module_user_config(cfg)
            template_mod = template_modules[i] if i < len(template_modules) else None

            if template_mod is not None:
                t.registration_surface_filename = template_mod.registration_surface_filename
                t.transducer_body_filename = template_mod.transducer_body_filename
                if template_mod.standoff_transform is not None:
                    t.standoff_transform = t.convert_transform(
                        np.array(template_mod.standoff_transform, dtype=float), template_mod.units,
                    )
                if template_mod.module_invert:
                    t.module_invert = list(template_mod.module_invert)

            # Resolve transform: template < device < explicit override
            transform = np.eye(4)
            if template_mod is not None:
                transform = t.convert_transform(np.array(template_mod.transform, dtype=float), template_mod.units)

            device_mod = device_modules_in_order[device_module_indices[i]] if device_cfg else None
            if device_mod is not None and device_mod.get("transform") is not None:
                transform = np.array(device_mod["transform"], dtype=float)

            if module_transforms is not None:
                transform = np.array(module_transforms[i], dtype=float)

            modules.append(TransformedTransducer.from_transducer(t, transform=transform))

        _validate_template_mesh_units(template, modules, arr_attrs)

        if device_cfg:
            if module_transforms is None:
                for recorded, entry in enumerate(device_modules_in_order):
                    transform = entry.get("transform")
                    if transform is not None and np.any(np.asarray(transform)[:3, 3]):
                        _recorded_module_units(recorded, device_module_origins, modules)
            if any(device_attrs.get(key) for key in ("registration_surface_filename", "transducer_body_filename")):
                mesh_units = _recorded_module_units(0, device_module_origins, modules)
                if getunitconversion(mesh_units, modules[0].units) != 1:
                    raise ValueError("Cannot inherit device mesh references across different units.")

        if "standoff_transform" in arr_attrs:
            st = arr_attrs["standoff_transform"]
            st = np.eye(4) if st is None else np.array(st, dtype=float)
            if st.shape != (4, 4):
                raise ValueError("standoff_transform must be a 4x4 matrix.")
            if "standoff_transform" in device_attrs:
                if np.any(st[:3, 3]):
                    units = _recorded_module_units(0, device_module_origins, modules)
                    st = modules[0].convert_transform(st, units)
            elif template is not None:
                if template_modules:
                    st = modules[0].convert_transform(st, template_modules[0].units)
                elif np.any(st[:3, 3]):
                    raise ValueError("Cannot infer standoff units from a template without modules.")
            arr_attrs["standoff_transform"] = st

        return cls(id=resolved_id, name=resolved_name, modules=modules, attrs=arr_attrs)

    @classmethod
    def get_connected(
        cls,
        interface=None,
        db=None,
        arr_id: str | None = None,
        arr_name: str | None = None,
        module_transforms: Sequence[np.ndarray] | None = None,
        use_default_template: bool = True,
    ) -> TransducerArray:
        """Read ``user_config`` from every connected TX module and build a :class:`TransducerArray`.

        If the lead module's ``user_config`` contains a nonempty ``device``
        block, it is validated before template selection: the number of
        modules listed must match the
        number of connected modules, and the recorded base58 ``hwid`` values
        must match the reported HWID set when any expected HWIDs are recorded.
        Without expected HWIDs, only the module count is checked. A mismatch
        raises :class:`DeviceConfigMismatchError`. When the ``device`` block
        carries a ``"template"`` field, that template id is preferred for the
        ``db`` lookup over the default ``(n_modules, freq)`` mapping below.

        Otherwise, picks a default template based on the number of connected
        modules and the per-module ``freq`` value (which must agree across
        modules when more than one is connected). The mapping is:

        ====================== =====================
        ``(n_modules, freq)``  template id
        ====================== =====================
        ``(1, 155)``           ``openlifu_1x155``
        ``(1, 400)``           ``openlifu_1x400``
        ``(2, 155)``           ``openlifu_2x155``
        ``(2, 400)``           ``openlifu_2x400``
        ====================== =====================

        When ``db`` is provided, the template (with its meshes) is loaded
        from the database via ``db.load_transducer(template_id, convert_array=False)``.
        If no database is provided (or the lookup fails) and
        ``use_default_template`` is ``True``, a meshless fallback template
        is constructed from the transforms embedded in this module, without
        mesh filenames. The 2x155 fallback uses stand-in geometry from the
        2x180 EVT1 template.

        Args:
            interface: an :py:class:`openlifu_sdk.io.LIFUInterface`-like
                object exposing ``txdevice.get_tx_module_count()`` and
                ``txdevice.read_config(module=i)``. A fresh
                :py:class:`LIFUInterface` is constructed when omitted
                (requires ``openlifu_sdk`` to be installed). An interface
                created here is closed on success or failure; an injected
                interface remains open.
            db: optional :py:class:`openlifu.db.Database` used to load the
                template by id (so the resulting array references the
                database's mesh files).
            arr_id: optional explicit override for the resulting array id.
            arr_name: optional explicit override for the resulting array name.
            module_transforms: optional explicit per-module 4x4 transforms
                (e.g. from a per-module calibration step) that override
                both the template and any device-config transforms.
            use_default_template: when ``True`` (default), fall back to a
                meshless embedded template if no database template can be
                found. ``False`` skips only the embedded fallback: a database
                template can still be used, or construction can proceed
                without a template.

        Returns:
            A :class:`TransducerArray` representing the connected device.
        """
        owns_interface = interface is None
        if owns_interface:
            try:
                from openlifu_sdk.io import LIFUInterface
            except ModuleNotFoundError as exc:
                if exc.name != "openlifu_sdk":
                    raise
                raise ImportError(
                    "openlifu_sdk is required to auto-create a LIFUInterface; "
                    "install it or pass an explicit `interface=` argument."
                ) from exc
            interface = LIFUInterface()

        try:
            txdevice = interface.txdevice
            count = int(txdevice.get_tx_module_count())
            if count <= 0:
                raise RuntimeError("No TX modules are connected.")

            user_configs: list[dict] = []
            for i in range(count):
                cfg = txdevice.read_config(module=i)
                if cfg is None:
                    raise RuntimeError(f"Failed to read user_config from module {i}.")
                user_configs.append(json.loads(cfg.get_json_str()))

            # All connected modules must report the same frequency for the
            # template lookup to be unambiguous.
            freqs = {c.get("freq") for c in user_configs}
            if len(freqs) > 1:
                raise ValueError(
                    f"Connected modules have mismatched frequencies: "
                    f"{sorted(f for f in freqs if f is not None)}"
                )
            freq = next(iter(freqs)) if freqs else None

            # Validate recorded identity before loading its template.
            device_cfg = user_configs[0].get("device") or None
            device_template_id: str | None = None
            if device_cfg:
                _validate_device_config_against_connected(device_cfg, user_configs)
                tid = device_cfg.get("template")
                if isinstance(tid, str) and tid:
                    device_template_id = tid

            # Resolve a template: prefer db lookup, fall back to embedded transforms.
            template: TransducerArray | None = None
            template_id: str | None = device_template_id
            if template_id is None and freq is not None:
                template_id = _DEFAULT_TEMPLATE_IDS.get((count, int(freq)))
            if template_id is not None:
                if db is not None:
                    try:
                        loaded = db.load_transducer(template_id, convert_array=False)
                    except Exception:  # pylint: disable=broad-exception-caught
                        # The optional database is only a source of template geometry.
                        # If it cannot supply one, use the configured fallback below.
                        loaded = None
                    if isinstance(loaded, TransducerArray):
                        template = loaded
                if template is None and use_default_template and template_id in _DEFAULT_TEMPLATE_DATA:
                    template = _build_meshless_default_template(template_id)

            arr = cls.from_module_user_configs(
                user_configs,
                template=template,
                module_transforms=module_transforms,
                arr_id=arr_id,
                arr_name=arr_name,
            )

            # Callers use this warning to ask about database overwrites.
            if db is not None:
                try:
                    known_ids = list(db.get_transducer_ids() or [])
                except Exception:  # pylint: disable=broad-exception-caught
                    known_ids = []
                if arr.id in known_ids:
                    try:
                        db_arr = db.load_transducer(arr.id, convert_array=False)
                    except Exception:  # pylint: disable=broad-exception-caught
                        db_arr = None
                    if isinstance(db_arr, TransducerArray) and not arrays_structurally_equal(arr, db_arr):
                        warnings.warn(
                            f"Connected transducer '{arr.id}' differs from the version "
                            f"stored in the database. The database version was not used.",
                            stacklevel=2,
                        )

            return arr
        finally:
            if owns_interface:
                interface.close()

    def to_device_config(self) -> dict:
        """Serialize array-level info to a ``device`` dict for the lead module's user_config.

        Captures the array ``id``, ``name``, and ``attrs`` (mesh filenames and
        array-level ``standoff_transform``) along with per-module ``hwid`` +
        ``transform`` entries. Mesh files themselves are not stored; consumers
        must combine this with a template :class:`TransducerArray` (which
        provides the mesh files via :py:attr:`Transducer.registration_surface_filename`
        / :py:attr:`Transducer.transducer_body_filename`) when reconstructing
        the array via :py:meth:`from_module_user_configs`.
        """
        attrs_serialized: dict = {}
        for k, v in self.attrs.items():
            attrs_serialized[k] = v.tolist() if isinstance(v, np.ndarray) else copy.deepcopy(v)
        modules_entries: list[dict] = []
        for m in self.modules:
            entry = {
                "hwid": (m.attrs or {}).get("hwid"),
                "transform": np.array(m.transform).tolist(),
            }
            modules_entries.append(entry)
        return {
            "id": self.id,
            "name": self.name,
            "modules": modules_entries,
            "attrs": attrs_serialized,
        }

    @property
    def registration_surface_filename(self):
        if "registration_surface_filename" in self.attrs:
            return self.attrs["registration_surface_filename"]
        return None

    @registration_surface_filename.setter
    def registration_surface_filename(self, value):
        self.attrs["registration_surface_filename"] = value

    @property
    def transducer_body_filename(self):
        if "transducer_body_filename" in self.attrs:
            return self.attrs["transducer_body_filename"]
        return None

    @transducer_body_filename.setter
    def transducer_body_filename(self, value):
        self.attrs["transducer_body_filename"] = value
