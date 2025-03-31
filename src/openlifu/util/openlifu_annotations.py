from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated


@dataclass
class OpenLIFUFieldData:
    """
    A lightweight class representing a name and annotation for the fields of a
    dataclass. For instance, the Graph dataclass will have two fields that are
    now associated with an OpenLIFUFieldData class containing its meaning:

    ```python
    @dataclass
    class Graph:
        units: Annotated[str, OpenLIFUFieldData("Units", "The units of the graph")] = "mm"
        dim_names: Annotated[
            Tuple[str, str, str],
            OpenLIFUFieldData("Dimensions", "The name of the dimensions of the graph."),
        ] = ("lat", "ele", "ax")
    ```

    Using Annotated[] will *not* interfere with runtime behavior or type
    compatibility, and so fields will behave as though no annotation is present.
    """

    name : Annotated[str, "The name of the dataclass field."] = field(default="Placeholder description")
    description : Annotated[str, "The description of the dataclass field."] = field(default="Placeholder description.")
