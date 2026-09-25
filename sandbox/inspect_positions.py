"""Inspect spatial positions of P-lane items."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

p = Path(
    r"h:\.shortcut-targets-by-id\1ASyd1Nwdck9enxdSry1PT-WBv_AiDtOX\Design Inputs"
    r"\Generic_SW_IEC62304_FDA_Cyber_Dependency_Map_V13_20260814.drawio.xml"
)
root = ET.parse(p).getroot()
cells = {c.get("id"): c for c in root.iter("mxCell") if c.get("id")}

# Get positions for a set of ids
def pos(cell):
    g = cell.find("mxGeometry")
    if g is None:
        return None, None, None, None
    return (
        float(g.get("x") or 0),
        float(g.get("y") or 0),
        float(g.get("width") or 0),
        float(g.get("height") or 0),
    )

lane_ids = [
    "lane1", "P-01", "P-02", "P-03", "P-04", "P-05",
    "P-06", "P-07", "P-08", "P-09", "P-10",
]
print(f"{'id':<8} {'x':>8} {'y':>8} {'w':>6} {'h':>6}  right = x+w")
for cid in lane_ids:
    c = cells.get(cid)
    if c is None:
        print(f"{cid:<8}  (missing)")
        continue
    x, y, w, h = pos(c)
    right = x + w if x is not None else None
    print(f"{cid:<8} {x:>8.0f} {y:>8.0f} {w:>6.0f} {h:>6.0f}  {right:>8.0f}")
