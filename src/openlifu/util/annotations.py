from __future__ import annotations

from typing import Annotated, NamedTuple


class OpenLIFUFieldData(NamedTuple):
    """
    A lightweight named tuple representing a name and annotation for the fields
    of a dataclass. For example, the Graph dataclass may have fields associated
    with this type:

    ```python
    class Graph:
        units: Annotated[str, OpenLIFUFieldData("Units", "The units of the graph")] = "mm"
        dim_names: Annotated[
            Tuple[str, str, str],
            OpenLIFUFieldData("Dimensions", "The name of the dimensions of the graph."),
        ] = ("lat", "ele", "ax")
    ```

    Annotated[] does not interfere with runtime behavior or type compatibility.
    """

    name: Annotated[str | None, "The name of the dataclass field."]
    description: Annotated[str | None, "The description of the dataclass field."]
