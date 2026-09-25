"""Compare styles of candidate item cells."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

p = Path(
    r"h:\.shortcut-targets-by-id\1ASyd1Nwdck9enxdSry1PT-WBv_AiDtOX\Design Inputs"
    r"\Generic_SW_IEC62304_FDA_Cyber_Dependency_Map_V13_20260814.drawio.xml"
)
root = ET.parse(p).getroot()
cells = {c.get("id"): c for c in root.iter("mxCell") if c.get("id")}
for cid in [
    "n1",
    "63RGjASBpJfkZYCKqKnj-1",
    "wY7z-qioInoPhFkbO_Ok-2",
    "CY-11",
    "REL-10",
    "P-01",
]:
    c = cells.get(cid)
    if c is None:
        continue
    print(f"{cid}:")
    print(f"  style: {c.get('style')}")
    print()
