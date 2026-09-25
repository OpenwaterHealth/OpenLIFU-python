"""Investigate edge cases in the drawio parsing."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_sw_flowchart import DRAWIO, html_to_text, style_dict  # noqa: E402

root = ET.parse(DRAWIO).getroot()
cells = {c.get("id"): c for c in root.iter("mxCell") if c.get("id")}

print("--- CY-11 raw value ---")
print(cells["CY-11"].get("value"))
print()
print("--- CY-11 as text ---")
print(html_to_text(cells["CY-11"].get("value", "")))
print()
print("--- REL-10 raw value ---")
print(cells["REL-10"].get("value"))
print()
print("--- 'Other' category edges ---")
for c in cells.values():
    if c.get("edge") != "1":
        continue
    d = style_dict(c.get("style", ""))
    color = d.get("strokeColor", "").lower()
    dashed = d.get("dashed") == "1"
    if color == "#475569" and not dashed:
        continue
    if color == "#b45309" and dashed:
        continue
    if color == "#94a3b8" and dashed:
        continue
    src = c.get("source")
    tgt = c.get("target")
    print(f"  {c.get('id')}: {src} -> {tgt}")
    print(f"    style: {c.get('style')}")
    print(f"    src label: {html_to_text((cells.get(src) or ET.Element('x')).get('value') or src)[:80]}")
    print(f"    tgt label: {html_to_text((cells.get(tgt) or ET.Element('x')).get('value') or tgt)[:80]}")
