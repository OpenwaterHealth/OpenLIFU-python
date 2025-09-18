from __future__ import annotations

import json

from openlifu.util.types import PathLike
from openlifu.xdc.transducer import Transducer
from openlifu.xdc.transducerarray import TransducerArray


def load_transducer_from_file(transducer_filepath : PathLike, convert_array:bool = True) -> Transducer|TransducerArray:
    """Load a Transducer or TransducerArray from file, depending on the "type" field in the file.
    Note: the transducer object includes the relative path to the affiliated transducer model data. `get_transducer_absolute_filepaths`, should
    be used to obtain the absolute data filepaths based on the Database directory path.
    Args:
        transducer_filepath: path to the transducer json file
        convert_array: When enabled, if a TransducerArray is encountered then it is converted to a Transducer.
    Returns: a Transducer if the json file defines a Transducer, or if the json file defines a TransducerArray and convert_array is enabled.
        Otherwise a TransducerArray.
    """
    with open(transducer_filepath) as f:
        if not f:
            raise FileNotFoundError(f"Transducer file not found at: {transducer_filepath}")
        d = json.load(f)
    if "type"  in d and d["type"] == "TransducerArray":
        transducer = TransducerArray.from_dict(d)
        if convert_array:
            transducer = transducer.to_transducer()
    else:
        transducer = Transducer.from_file(transducer_filepath)
    return transducer
