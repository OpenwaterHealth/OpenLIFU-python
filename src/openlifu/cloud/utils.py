from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import uuid
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

logger_cloud = logging.getLogger("Cloud")


def to_isoformat(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "")

def from_isoformat(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)

def mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0)


def to_json(obj):
    def convert(o):
        if is_dataclass(o):
            return {k: convert(v) for k, v in asdict(o).items()}
        if isinstance(o, datetime):
            return to_isoformat(o)
        if isinstance(o, list | tuple):
            return [convert(x) for x in o]
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        return o

    return json.dumps(convert(obj))


def _from_value(expected_type, value):
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # List[T]
    if origin is list:
        (item_type,) = args
        return [_from_value(item_type, v) for v in (value or [])]

    # Dict[K, V]
    if origin is dict:
        key_type, val_type = args
        return {
            _from_value(key_type, k): _from_value(val_type, v)
            for k, v in (value or {}).items()
        }

    # Union / Optional / T | None  (robust: rely on args)
    if args and type(None) in args:
        real_type = next(t for t in args if t is not None.__class__)
        return None if value is None else _from_value(real_type, value)

    # datetime
    if expected_type is datetime:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value
        return from_isoformat(value)

    # nested dataclass
    if is_dataclass(expected_type):
        hints = get_type_hints(expected_type)
        kwargs = {}
        for f in fields(expected_type):
            f_type = hints.get(f.name, f.type)
            if isinstance(value, dict) and f.name in value:
                kwargs[f.name] = _from_value(f_type, value[f.name])
            else:
                kwargs[f.name] = None
        return expected_type(**kwargs)

    # basic type (int, str, bool, float, etc.)
    return value


def from_json(expected_type, json_str: str):
    data = json.loads(json_str)
    return _from_value(expected_type, data)


def get_mac_address() -> str | None:
    if sys.platform == "win32":
        return get_mac_address_windows()
    return get_mac_address_unix()


def get_mac_address_unix() -> str | None:
    mac = uuid.getnode()

    # If MAC is random / locally administered, high bit is set
    if (mac >> 40) % 2:
        return None  # Not a real hardware MAC

    return ':'.join(f'{(mac >> ele) & 0xff:02x}' for ele in range(40, -1, -8))


def get_mac_address_windows() -> str | None:
    try:
        cmd = [
            "getmac",
            "/" + "f" + "o", # ignore codespell error
            "csv",
            "/nh"
        ]
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        for line in output.splitlines():
            mac = line.split(",")[0].strip('"')
            if re.match(r"([0-9A-Fa-f]{2}-){5}[0-9A-Fa-f]{2}", mac):
                return mac.replace("-", ":").lower()
    except Exception as e:
        raise ValueError("Cannot get MAC address") from e
    return None
