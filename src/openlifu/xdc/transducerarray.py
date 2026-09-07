from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from openlifu.util.dict_conversion import DictMixin
from openlifu.util.units import getunitconversion
from openlifu.xdc import Transducer, TransformedTransducer


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
           keyed by ``hwid`` (falling back to positional matching when
           no matching HWID entry is found).
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
        count validation; an absent or empty block is valid.

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
        device_modules_in_order: list = []
        device_modules_by_hwid: dict = {}
        if device_cfg:
            _validate_device_config_against_connected(device_cfg, user_configs)
            resolved_id = device_cfg.get("id", resolved_id)
            resolved_name = device_cfg.get("name", resolved_name)
            for k, v in (device_cfg.get("attrs") or {}).items():
                arr_attrs[k] = copy.deepcopy(v)
            device_modules_in_order = list(device_cfg.get("modules") or [])
            device_modules_by_hwid = {
                m["hwid"]: m
                for m in device_modules_in_order
                if isinstance(m, dict) and m.get("hwid")
            }

        if arr_id is not None:
            resolved_id = arr_id
        if arr_name is not None:
            resolved_name = arr_name

        st = arr_attrs.get("standoff_transform")
        if st is not None and not isinstance(st, np.ndarray):
            arr_attrs["standoff_transform"] = np.array(st, dtype=float)

        template_modules: list = list(template.modules) if template is not None else []
        modules: list[TransformedTransducer] = []
        for i, cfg in enumerate(user_configs):
            t = Transducer.from_module_user_config(cfg)
            template_mod = template_modules[i] if i < len(template_modules) else None

            if template_mod is not None:
                t.registration_surface_filename = template_mod.registration_surface_filename
                t.transducer_body_filename = template_mod.transducer_body_filename
                if template_mod.standoff_transform is not None:
                    t.standoff_transform = np.array(template_mod.standoff_transform, dtype=float)
                if template_mod.module_invert:
                    t.module_invert = list(template_mod.module_invert)

            # Resolve transform: template < device < explicit override
            transform = np.eye(4)
            if template_mod is not None:
                transform = np.array(template_mod.transform, dtype=float)

            hwid = cfg.get("hwid")
            device_mod: dict | None = None
            if hwid and hwid in device_modules_by_hwid:
                device_mod = device_modules_by_hwid[hwid]
            elif device_modules_in_order and i < len(device_modules_in_order):
                candidate = device_modules_in_order[i]
                if isinstance(candidate, dict):
                    device_mod = candidate
            if device_mod is not None and device_mod.get("transform") is not None:
                transform = np.array(device_mod["transform"], dtype=float)

            if module_transforms is not None:
                transform = np.array(module_transforms[i], dtype=float)

            modules.append(TransformedTransducer.from_transducer(t, transform=transform))

        return cls(id=resolved_id, name=resolved_name, modules=modules, attrs=arr_attrs)

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
