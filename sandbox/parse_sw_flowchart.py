"""Parse the SW dependency drawio XML and update the SW Flowchart.xlsx."""
from __future__ import annotations

import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from html import unescape
from pathlib import Path

from openpyxl import load_workbook

DRAWIO = Path(
    r"h:\.shortcut-targets-by-id\1ASyd1Nwdck9enxdSry1PT-WBv_AiDtOX\Design Inputs"
    r"\Generic_SW_IEC62304_FDA_Cyber_Dependency_Map_V13_20260814.drawio.xml"
)
XLSX = Path(
    r"h:\.shortcut-targets-by-id\1ASyd1Nwdck9enxdSry1PT-WBv_AiDtOX\Design Inputs"
    r"\SW Flowchart.xlsx"
)

ITEM_ID = re.compile(r"^[A-Z]{1,3}-[A-Z0-9]{1,4}[a-z]?$")
# Match a leading item token like "CY-12" possibly preceded by opening HTML tags
LEADING_ID = re.compile(r"^\s*(?:<[^>]+>\s*)*([A-Z]{1,3}-[A-Z0-9]{1,4}[a-z]?)\b")

# Manual dependency overrides for edges that are missing / unfinished in the
# drawio source. Each entry adds the listed predecessors to the target's
# dependency list.
MANUAL_DEPS: dict[str, list[str]] = {
    "CY-11": ["CY-04"],
}


def style_dict(style: str | None) -> dict[str, str]:
    d: dict[str, str] = {}
    for kv in (style or "").split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k] = v
    return d


def classify_edge(style: str | None) -> str:
    """Return 'primary', 'feedback', or 'scales'.

    Legend from the diagram:
      solid grey (#475569, not dashed) = primary predecessor
      amber dashed (#b45309, dashed)   = controlled iteration / feedback
      grey dotted (#94a3b8, dashed)    = scales the depth of (not a predecessor)

    Hand-added edges without an explicit strokeColor inherit drawio defaults;
    inspection of the file shows these are all intended as primary predecessors,
    so we treat them as such.
    """
    d = style_dict(style)
    color = d.get("strokeColor", "").lower()
    dashed = d.get("dashed") == "1"
    if color == "#b45309" and dashed:
        return "feedback"
    if color == "#94a3b8" and dashed:
        return "scales"
    return "primary"


TAG_RE = re.compile(r"<[^>]+>")
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def html_to_text(s: str) -> str:
    if not s:
        return ""
    s = BR_RE.sub("\n", s)
    s = TAG_RE.sub("", s)
    return unescape(s).strip()


def _is_decorative(line: str) -> bool:
    """Return True if line is only decorative glyphs (★, ⚡, etc.)."""
    stripped = line.strip()
    if not stripped:
        return True
    return len(stripped) <= 2 and not any(ch.isalnum() for ch in stripped)


def parse_node_label(item_id: str, raw_value: str) -> tuple[str, str]:
    """Return (title, description) from a node's value.

    Node labels are usually authored as:
        <b>ID</b><br><b>Title</b><br>Description...
    but some hand-edited nodes use:
        <div><b>ID</b> <span>Title text</span></b></div><div>Description...</div>
    We handle both.
    """
    # Convert <div>...</div> boundaries to line breaks so downstream logic sees
    # one logical line per div. Drop the outer div tags themselves.
    v = raw_value or ""
    v = re.sub(r"</div>\s*<div[^>]*>", "\n", v)
    v = re.sub(r"</?div[^>]*>", "", v)
    text = html_to_text(v)
    # Normalise NBSP to space
    text = text.replace("\xa0", " ")
    lines = [ln.strip() for ln in text.splitlines()]
    # Drop leading blank lines
    while lines and not lines[0]:
        lines.pop(0)
    if not lines:
        return "", ""
    first = lines[0]
    # Case A: first line is exactly the item ID -> title is the next
    # meaningful line (skip decorative-glyph-only lines such as REL-10's "★")
    if first == item_id:
        lines.pop(0)
        while lines and (not lines[0] or _is_decorative(lines[0])):
            lines.pop(0)
        if not lines:
            return "", ""
        title = lines[0]
        body = lines[1:]
    # Case B: first line starts with the ID followed by the title text
    #         e.g. "CY-11 Cybersecurity test plan"
    elif first.startswith(item_id):
        title = first[len(item_id):].lstrip(" \t:-—").strip()
        body = lines[1:]
    else:
        title = first
        body = lines[1:]
    body_lines: list[str] = []
    for ln in body:
        if ln:
            body_lines.append(ln)
        elif body_lines and not body_lines[-1].endswith("."):
            body_lines[-1] = body_lines[-1] + "."
    description = " ".join(body_lines).strip()
    return title, description


def main() -> None:
    root = ET.parse(DRAWIO).getroot()
    cells = {c.get("id"): c for c in root.iter("mxCell") if c.get("id")}

    # Discover items: any vertex whose cell id matches ITEM_ID, OR whose label
    # starts with an ID token AND whose style is a rounded item box (not a
    # note / text annotation).
    items: dict[str, tuple[str, str]] = {}
    cell_to_item: dict[str, str] = {}  # cell id -> item id (alias map)
    for cid, cell in cells.items():
        if cell.get("vertex") != "1":
            continue
        item_id: str | None = None
        if ITEM_ID.match(cid):
            item_id = cid
        else:
            style = cell.get("style", "") or ""
            if "rounded=1" in style:
                m = LEADING_ID.match(cell.get("value", "") or "")
                if m:
                    item_id = m.group(1)
        if item_id is None:
            continue
        title, desc = parse_node_label(item_id, cell.get("value", ""))
        items[item_id] = (title, desc)
        cell_to_item[cid] = item_id

    # Edges: primary predecessors only
    deps: dict[str, list[str]] = defaultdict(list)
    edge_counts = {"primary": 0, "feedback": 0, "scales": 0}
    dropped_edges: list[tuple[str, str, str]] = []
    for cell in cells.values():
        if cell.get("edge") != "1":
            continue
        cat = classify_edge(cell.get("style", ""))
        edge_counts[cat] += 1
        src_raw = cell.get("source")
        tgt_raw = cell.get("target")
        src = cell_to_item.get(src_raw, src_raw)
        tgt = cell_to_item.get(tgt_raw, tgt_raw)
        if cat != "primary":
            continue
        if src in items and tgt in items:
            if src not in deps[tgt]:
                deps[tgt].append(src)
        else:
            dropped_edges.append((cell.get("id", "?"), str(src_raw), str(tgt_raw)))

    # Apply manual dependency overrides
    manual_applied: list[tuple[str, str]] = []
    for tgt, preds in MANUAL_DEPS.items():
        if tgt not in items:
            continue
        for src in preds:
            if src in items and src not in deps[tgt]:
                deps[tgt].append(src)
                manual_applied.append((src, tgt))

    # Lane order from the diagram's serpentine reading:
    # Lane 1 = Planning (P), Lane 2 = IEC 62304 development (SW / SW-L / DR),
    # Lane 3 = Cybersecurity (CY), Lane 4 = V&V (VV), Lane 5 = Release (REL).
    LANE_ORDER = {"P": 1, "SW": 2, "DR": 2, "CY": 3, "VV": 4, "REL": 5}

    def sort_key(item_id: str) -> tuple[int, str, int, str]:
        m = re.match(r"^([A-Z]+)-([A-Z]?)(\d+)([a-z]?)$", item_id)
        if not m:
            return (99, item_id, 0, "")
        prefix = m.group(1)
        letter_suffix = m.group(2)
        number = int(m.group(3))
        trailing = m.group(4)
        lane = LANE_ORDER.get(prefix, 99)
        return (lane, prefix + letter_suffix, number, trailing)

    ordered_ids = sorted(items.keys(), key=sort_key)

    # ---- Linear index: Kahn's topological sort, alphanumeric tiebreak ----
    # Only consider dependencies among discovered items (defensive).
    remaining_deps = {iid: {d for d in deps[iid] if d in items} for iid in items}
    successors: dict[str, set[str]] = defaultdict(set)
    for tgt, preds in remaining_deps.items():
        for src in preds:
            successors[src].add(tgt)

    linear_index: dict[str, int] = {}
    ready = sorted(
        [iid for iid, preds in remaining_deps.items() if not preds],
        key=sort_key,
    )
    idx = 1
    while ready:
        current = ready.pop(0)
        linear_index[current] = idx
        idx += 1
        newly_ready: list[str] = []
        for succ in successors[current]:
            remaining_deps[succ].discard(current)
            if not remaining_deps[succ] and succ not in linear_index and succ not in ready:
                newly_ready.append(succ)
        if newly_ready:
            ready = sorted(ready + newly_ready, key=sort_key)

    if len(linear_index) != len(items):
        missing = [iid for iid in items if iid not in linear_index]
        print(
            f"WARNING: topological sort incomplete ({len(linear_index)}/{len(items)}). "
            f"Cycle or unresolved deps involving: {missing}"
        )
        # Append remaining alphabetically so the sheet still gets an index.
        for iid in sorted(missing, key=sort_key):
            linear_index[iid] = idx
            idx += 1

    print(f"Items: {len(items)}   Edges by category: {edge_counts}")
    if dropped_edges:
        print(f"Dropped {len(dropped_edges)} primary edges with unresolved endpoints:")
        for eid, s, t in dropped_edges:
            print(f"  {eid}: {s} -> {t}")
    if manual_applied:
        print(f"Manual dependency overrides applied ({len(manual_applied)}):")
        for s, t in manual_applied:
            print(f"  {s} -> {t}")
    print()
    print(f"{'#':>3}  {'ID':<8} {'Title':<50} Dependencies")
    print("-" * 104)
    for iid in sorted(items.keys(), key=lambda i: linear_index[i]):
        title, _ = items[iid]
        d = ", ".join(sorted(deps[iid], key=sort_key))
        print(f"{linear_index[iid]:>3}  {iid:<8} {title[:48]:<50} {d}")

    # ---------------- Update xlsx ----------------
    backup = XLSX.with_name(XLSX.stem + ".backup.xlsx")
    shutil.copyfile(XLSX, backup)
    print(f"\nBackup saved: {backup}")

    wb = load_workbook(XLSX)
    ws = wb.active

    # Ensure a "Linear Index" column exists as column A. Insert one if it's
    # not already there, preserving existing data.
    header = [c.value for c in ws[1]]
    if "Linear Index" not in header:
        ws.insert_cols(1)
        ws.cell(1, 1).value = "Linear Index"
        header = [c.value for c in ws[1]]

    # Preserve existing values for the columns already filled in by the user
    # (their edits to Title/Description are hand-tuned) and for Completed and
    # File Link. Only fill / overwrite Dependencies and Linear Index from the
    # diagram. For new IDs not yet in the sheet, add rows with the extracted
    # title/description.

    col_idx = {name: i + 1 for i, name in enumerate(header)}
    index_col = col_idx["Linear Index"]
    id_col = col_idx["ID"]
    title_col = col_idx["Title"]
    desc_col = col_idx["Description"]
    completed_col = col_idx["Completed"]
    filelink_col = col_idx["File Link"]
    dep_col = col_idx["Dependencies"]

    existing: dict[str, int] = {}
    last_data_row = 1
    for r in range(2, ws.max_row + 1):
        val = ws.cell(r, id_col).value
        if val:
            existing[str(val).strip()] = r
            last_data_row = r

    # Update existing rows: touch Dependencies and Linear Index only
    for iid, row in existing.items():
        if iid in deps:
            ws.cell(row, dep_col).value = ", ".join(sorted(deps[iid], key=sort_key))
        if iid in linear_index:
            ws.cell(row, index_col).value = linear_index[iid]

    # Append missing rows in linear-index order
    next_row = last_data_row + 1
    added = 0
    for iid in sorted(items.keys(), key=lambda i: linear_index[i]):
        if iid in existing:
            continue
        title, description = items[iid]
        ws.cell(next_row, index_col).value = linear_index[iid]
        ws.cell(next_row, id_col).value = iid
        ws.cell(next_row, title_col).value = title
        ws.cell(next_row, desc_col).value = description
        ws.cell(next_row, completed_col).value = False
        ws.cell(next_row, filelink_col).value = None
        ws.cell(next_row, dep_col).value = ", ".join(
            sorted(deps[iid], key=sort_key)
        )
        next_row += 1
        added += 1

    print(
        f"Existing rows updated: {len(existing)}   New rows appended: {added}"
    )

    wb.save(XLSX)
    print(f"Saved: {XLSX}")


if __name__ == "__main__":
    main()
