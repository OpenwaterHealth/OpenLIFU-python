import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict, is_dataclass, fields
from typing import get_origin, get_args, Optional
import json


def to_isoformat(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "")


def mtime(path: Path) -> datetime:
    stat = os.stat(path)
    return datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).replace(microsecond=0)


def to_json(obj):
    def convert(o):
        if is_dataclass(o):
            return {k: convert(v) for k, v in asdict(o).items()}
        if isinstance(o, datetime):
            return to_isoformat(o)
        if isinstance(o, (list, tuple)):
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
        item_type = args[0]
        return [_from_value(item_type, v) for v in value]

    # Dict[str, T]
    if origin is dict:
        value_type = args[1]
        return {k: _from_value(value_type, v) for k, v in value.items()}

    # Optional[T]  (Union[T, None])
    if origin is not None and type(None) in args:
        real_type = next(t for t in args if t is not type(None))
        return None if value is None else _from_value(real_type, value)

    # datetime
    if expected_type is datetime:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)

    # nested dataclass
    if is_dataclass(expected_type):
        kwargs = {}
        for f in fields(expected_type):
            if f.name in value:
                kwargs[f.name] = _from_value(f.type, value[f.name])
            else:
                kwargs[f.name] = None
        return expected_type(**kwargs)

    # basic type (int, str, bool, float, etc.)
    return value


def from_json(expected_type, json_str: str):
    data = json.loads(json_str)
    return _from_value(expected_type, data)


def get_mac_address() -> Optional[str]:
    mac = uuid.getnode()

    # If MAC is random / locally administered, high bit is set
    if (mac >> 40) % 2:
        return None  # Not a real hardware MAC

    return ':'.join(f'{(mac >> ele) & 0xff:02x}' for ele in range(40, -1, -8))