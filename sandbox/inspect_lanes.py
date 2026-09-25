"""Dump all items with their x,y positions and lane assignment."""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_sw_flowchart import DRAWIO, ITEM_ID, LEADING_ID

root = ET.parse(DRAWIO).getroot()
cells = {c.get("id"): c for c in root.iter("mxCell") if c.get("id")}


def geom(cell):
    g = cell.find("mxGeometry")
    if g is None:
        return None
    return (
        float(g.get("x") or 0),
        float(g.get("y") or 0),
        float(g.get("width") or 0),
        float(g.get("height") or 0),
    )


# Find lane header cells (look for "lane" IDs or labels starting with "1.", "2.", ...)
print("=== lane headers (nodes named lane*/labels starting '<digit>.') ===")
for cid, c in cells.items():
    val = c.get("value") or ""
    if cid.startswith("lane") or re.match(r"^\s*<[^>]*>\s*\d\.", val):
        g = geom(c)
        if g:
            x, y, w, h = g
            print(f"  {cid:<10}  x={x:>6.0f} y={y:>6.0f} w={w:>5.0f} h={h:>5.0f}  value={val[:100]!r}")

print()
print("=== all items with position, sorted by y then x ===")
items = []
for cid, c in cells.items():
    if c.get("vertex") != "1":
        continue
    item_id = None
    if ITEM_ID.match(cid):
        item_id = cid
    else:
        style = c.get("style", "") or ""
        if "rounded=1" in style:
            m = LEADING_ID.match(c.get("value", "") or "")
            if m:
                item_id = m.group(1)
    if not item_id:
        continue
    g = geom(c)
    if g:
        x, y, w, h = g
        items.append((item_id, x, y, w, h))

items.sort(key=lambda t: (t[2], t[1]))
for iid, x, y, w, h in items:
    print(f"  {iid:<8} x={x:>6.0f} y={y:>6.0f} w={w:>5.0f} h={h:>5.0f}")
