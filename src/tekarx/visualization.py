# ruff: noqa: E501
"""Memory-safe, self-contained visualization of sampled patient-drug graphs."""

from __future__ import annotations

import html
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}
REQUIRED_ARRAYS = (
    "patient_primaryid",
    "patient_y",
    "patient_split_id",
    "edge_patient_index",
    "edge_drug_index",
)


class GraphVisualizationError(RuntimeError):
    """Raised when a graph cannot be safely sampled or rendered."""


@dataclass(frozen=True)
class GraphVisualizationRecord:
    """Summary of a rendered graph sample."""

    output_path: str
    array_manifest_path: str
    split: str
    layout: str
    requested_patients: int
    rendered_patients: int
    rendered_drugs: int
    rendered_edges: int
    serious_patients: int
    nonserious_patients: int
    graph_patient_nodes: int
    graph_drug_nodes: int
    graph_edges: int
    seed: int


def visualize_graph(
    *,
    data_dir: Path,
    graph_dir: Path | None = None,
    split: str = "validation",
    layout: str = "2d",
    patients: int = 100,
    top_drugs: int = 50,
    seed: int = 42,
    output: Path | None = None,
) -> GraphVisualizationRecord:
    """Render a bounded patient-drug sample as standalone interactive HTML.

    Only the topology, split, target, patient-ID, and drug-metadata arrays are
    opened. All NumPy arrays remain memory-mapped, so a full graph does not need
    to fit in RAM.
    """
    if split not in SPLIT_IDS:
        raise ValueError(f"split must be one of {tuple(SPLIT_IDS)}")
    if layout not in ("2d", "3d"):
        raise ValueError("layout must be '2d' or '3d'")
    if not 1 <= patients <= 500:
        raise ValueError("patients must be between 1 and 500")
    if not 1 <= top_drugs <= 250:
        raise ValueError("top_drugs must be between 1 and 250")

    data_root = Path(data_dir).resolve()
    manifest_path = _resolve_array_manifest(data_root, graph_dir)
    manifest = _read_manifest(manifest_path)
    arrays = {
        name: _load_array(manifest_path, manifest, name)
        for name in REQUIRED_ARRAYS
    }
    drug_x = _load_array(manifest_path, manifest, "drug_x", required=False)

    patient_count = int(arrays["patient_y"].shape[0])
    if arrays["patient_primaryid"].shape != (patient_count,):
        raise GraphVisualizationError("patient ID and target arrays have different lengths")
    if arrays["patient_split_id"].shape != (patient_count,):
        raise GraphVisualizationError("patient split and target arrays have different lengths")
    if arrays["edge_patient_index"].shape != arrays["edge_drug_index"].shape:
        raise GraphVisualizationError("patient and drug edge arrays have different lengths")

    selected = _sample_patient_indices(
        arrays["patient_split_id"],
        arrays["patient_y"],
        split_id=SPLIT_IDS[split],
        count=patients,
        seed=seed,
    )
    sampled_edges = _edges_for_patients(
        arrays["edge_patient_index"],
        arrays["edge_drug_index"],
        selected,
        manifest=manifest,
        split=split,
    )
    if not sampled_edges:
        raise GraphVisualizationError(f"sampled {split} patients have no drug edges")

    frequencies = Counter(drug for _patient, drug in sampled_edges)
    kept_drugs = {
        drug
        for drug, _degree in sorted(
            frequencies.items(), key=lambda item: (-item[1], item[0])
        )[:top_drugs]
    }
    kept_edges = sorted(
        {(patient, drug) for patient, drug in sampled_edges if drug in kept_drugs}
    )
    kept_patients = sorted({patient for patient, _drug in kept_edges})
    if not kept_patients:
        raise GraphVisualizationError("top-drug filtering removed every sampled patient")

    drug_metadata = _read_drug_metadata(manifest_path, manifest)
    drug_names = _drugcentral_names(data_root, kept_drugs, drug_metadata)
    patient_nodes = [
        {
            "id": f"p:{index}",
            "index": index,
            "primaryid": str(int(arrays["patient_primaryid"][index])),
            "serious": int(arrays["patient_y"][index]),
        }
        for index in kept_patients
    ]
    drug_nodes = []
    for index in sorted(kept_drugs, key=lambda value: (-frequencies[value], value)):
        metadata = drug_metadata.get(index, {})
        features = _describe_drug_features(drug_x, index)
        drug_nodes.append(
            {
                "id": f"d:{index}",
                "index": index,
                "label": drug_names.get(index) or str(metadata.get("node_label", f"Drug {index}")),
                "semantic_id": int(metadata.get("semantic_id", 0)),
                "kind": str(metadata.get("node_kind", "unknown")),
                "degree": frequencies[index],
                **features,
            }
        )
    edges = [
        {"source": f"p:{patient}", "target": f"d:{drug}"}
        for patient, drug in kept_edges
    ]

    default_name = "graph_visualization_3d.html" if layout == "3d" else "graph_visualization.html"
    output_path = Path(output or data_root / "processed" / default_name)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "source_manifest": str(manifest_path),
        "patients": patient_nodes,
        "drugs": drug_nodes,
        "edges": edges,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(_render_html(payload, layout=layout), encoding="utf-8")
    temporary.replace(output_path)

    counts = manifest.get("counts", {})
    record = GraphVisualizationRecord(
        output_path=str(output_path),
        array_manifest_path=str(manifest_path),
        split=split,
        layout=layout,
        requested_patients=patients,
        rendered_patients=len(patient_nodes),
        rendered_drugs=len(drug_nodes),
        rendered_edges=len(edges),
        serious_patients=sum(node["serious"] for node in patient_nodes),
        nonserious_patients=sum(1 - node["serious"] for node in patient_nodes),
        graph_patient_nodes=int(counts.get("patient_nodes", patient_count)),
        graph_drug_nodes=int(counts.get("drug_nodes", len(drug_metadata))),
        graph_edges=int(counts.get("patient_drug_edges", arrays["edge_patient_index"].shape[0])),
        seed=seed,
    )
    manifest_output = output_path.with_suffix(".json")
    manifest_output.write_text(
        json.dumps(
            {
                "dataset": "TekaRx sampled patient-drug graph visualization",
                "intended_use": "research exploration; not diagnosis",
                "sampling": "seeded, approximately class-balanced patient sample",
                "record": asdict(record),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return record


def _resolve_array_manifest(data_dir: Path, graph_dir: Path | None) -> Path:
    if graph_dir is not None:
        source = Path(graph_dir).resolve()
        candidates = (
            (source,) if source.is_file() else (
                source / "tekarx_graph_arrays" / "manifest.json",
                source / "manifest.json",
            )
        )
    else:
        checkpoints = data_dir / "processed" / "graph_visualization_checkpoints"
        candidates_list: list[Path] = []
        if checkpoints.is_dir():
            candidates_list.extend(
                directory / "tekarx_graph_arrays" / "manifest.json"
                for directory in sorted(checkpoints.iterdir(), reverse=True)
                if directory.is_dir() and not directory.name.startswith(".")
            )
        candidates_list.append(data_dir / "processed" / "tekarx_graph_arrays" / "manifest.json")
        candidates = tuple(candidates_list)
    for candidate in candidates:
        if candidate.is_file() and _has_required_arrays(candidate):
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise GraphVisualizationError(f"no complete visualization graph found; searched: {searched}")


def _has_required_arrays(manifest_path: Path) -> bool:
    try:
        manifest = _read_manifest(manifest_path)
        for name in REQUIRED_ARRAYS:
            metadata = manifest.get("arrays", {}).get(name, {})
            relative = metadata.get("path")
            if not isinstance(relative, str) or not (manifest_path.parent / relative).is_file():
                return False
        metadata_path = manifest.get("drug_metadata_path")
        return isinstance(metadata_path, str) and (manifest_path.parent / metadata_path).is_file()
    except GraphVisualizationError:
        return False


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphVisualizationError(f"cannot read graph array manifest: {path}") from exc
    if manifest.get("format") != "tekarx.memmap_graph":
        graph_format = manifest.get("format")
        raise GraphVisualizationError(f"unsupported graph manifest format: {graph_format!r}")
    return manifest


def _load_array(
    manifest_path: Path,
    manifest: dict[str, Any],
    name: str,
    *,
    required: bool = True,
) -> np.ndarray | None:
    metadata = manifest.get("arrays", {}).get(name)
    if not isinstance(metadata, dict):
        if required:
            raise GraphVisualizationError(f"graph manifest has no {name!r} array")
        return None
    relative = metadata.get("path")
    if not isinstance(relative, str):
        raise GraphVisualizationError(f"graph array {name!r} has no path")
    root = manifest_path.parent.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        if not required:
            return None
        raise GraphVisualizationError(f"missing graph array {name!r}: {path}")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise GraphVisualizationError(f"cannot memory-map graph array {name!r}: {path}") from exc
    if str(array.dtype) != metadata.get("dtype") or list(array.shape) != metadata.get("shape"):
        raise GraphVisualizationError(f"graph array {name!r} does not match its manifest")
    return array


def _sample_patient_indices(
    split_ids: np.ndarray,
    labels: np.ndarray,
    *,
    split_id: int,
    count: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    targets = {1: count // 2, 0: count - count // 2}
    selected: dict[int, list[int]] = {0: [], 1: []}
    seen: set[int] = set()
    maximum_draws = min(max(int(split_ids.shape[0]) * 2, 50_000), 5_000_000)
    draws = 0
    while sum(len(values) for values in selected.values()) < count and draws < maximum_draws:
        batch = rng.integers(0, split_ids.shape[0], size=min(4096, maximum_draws - draws))
        draws += int(batch.size)
        for raw_index in batch:
            index = int(raw_index)
            if index in seen or int(split_ids[index]) != split_id:
                continue
            label = int(labels[index])
            if label not in (0, 1) or len(selected[label]) >= targets[label]:
                continue
            seen.add(index)
            selected[label].append(index)
            if sum(len(values) for values in selected.values()) >= count:
                break
    chosen = selected[0] + selected[1]
    if len(chosen) < count:
        for start in range(0, split_ids.shape[0], 1_000_000):
            stop = min(start + 1_000_000, split_ids.shape[0])
            candidates = np.flatnonzero(np.asarray(split_ids[start:stop]) == split_id) + start
            for raw_index in candidates:
                index = int(raw_index)
                if index in seen:
                    continue
                seen.add(index)
                chosen.append(index)
                if len(chosen) >= count:
                    break
            if len(chosen) >= count:
                break
    if not chosen:
        raise GraphVisualizationError("selected graph split has no patients")
    return sorted(chosen[:count])


def _edges_for_patients(
    edge_patients: np.ndarray,
    edge_drugs: np.ndarray,
    patients: list[int],
    *,
    manifest: dict[str, Any],
    split: str,
) -> list[tuple[int, int]]:
    offsets = manifest.get("edge_order", {}).get("split_offsets", {}).get(split)
    start, stop = (int(offsets[0]), int(offsets[1])) if offsets else (0, edge_patients.shape[0])
    patient_view = edge_patients[start:stop]
    drug_view = edge_drugs[start:stop]
    result: list[tuple[int, int]] = []
    for patient in patients:
        left = int(np.searchsorted(patient_view, patient, side="left"))
        right = int(np.searchsorted(patient_view, patient, side="right"))
        result.extend((patient, int(drug)) for drug in drug_view[left:right])
    return result


def _read_drug_metadata(
    manifest_path: Path, manifest: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    relative = manifest.get("drug_metadata_path")
    if not isinstance(relative, str):
        raise GraphVisualizationError("graph manifest has no drug_metadata_path")
    path = (manifest_path.parent / relative).resolve()
    if not path.is_file():
        raise GraphVisualizationError(f"missing drug metadata: {path}")
    table = pq.read_table(path, columns=["node_index", "semantic_id", "node_label", "node_kind"])
    return {
        int(row["node_index"]): row
        for row in table.to_pylist()
    }


def _drugcentral_names(
    data_dir: Path,
    indices: set[int],
    metadata: dict[int, dict[str, Any]],
) -> dict[int, str]:
    structures = data_dir / "interim" / "drugcentral" / "structures.parquet"
    if not structures.is_file():
        return {}
    wanted = {
        str(metadata[index].get("semantic_id")): index
        for index in indices
        if metadata.get(index, {}).get("node_kind") == "mapped"
    }
    if not wanted:
        return {}
    table = pq.read_table(structures, columns=["id", "name"])
    result: dict[int, str] = {}
    for row in table.to_pylist():
        index = wanted.get(str(row["id"]))
        if index is not None and row.get("name"):
            result[index] = str(row["name"])
    return result


def _describe_drug_features(drug_x: np.ndarray | None, index: int) -> dict[str, Any]:
    if drug_x is None or index >= drug_x.shape[0] or drug_x.shape[1] < 2:
        return {"ror_z": None, "boxed_warning": False, "atc": "?"}
    values = np.asarray(drug_x[index])
    atc = "?"
    if values.shape[0] >= 28 and float(np.max(values[2:28])) > 0.5:
        atc = chr(ord("A") + int(np.argmax(values[2:28])))
    return {
        "ror_z": round(float(values[0]), 4),
        "boxed_warning": bool(values[1] >= 0.5),
        "atc": atc,
    }


def _render_html(payload: dict[str, Any], *, layout: str) -> str:
    if layout == "3d":
        return _render_3d_html(payload)
    return _render_2d_html(payload)


def _render_2d_html(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = f"TekaRx {payload['split'].title()} Patient–Drug Graph"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: dark; font-family: Inter,Segoe UI,sans-serif; }}
body {{ margin:0; background:#0b1020; color:#e5e7eb; }}
header {{ padding:18px 24px; background:#111827; position:sticky; top:0; z-index:5; }}
h1 {{ margin:0 0 6px; font-size:22px; }} .sub {{ color:#9ca3af; font-size:13px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:16px; margin-top:12px; align-items:center; }}
label {{ font-size:13px; }} input[type=range] {{ vertical-align:middle; }}
#canvas {{ overflow:auto; height:calc(100vh - 145px); }}
svg {{ min-width:1100px; background:radial-gradient(circle at center,#172033,#0b1020 65%); }}
.edge {{ stroke:#64748b; stroke-width:1; }}
.patient {{ stroke:#fff; stroke-width:1.2; cursor:pointer; }}
.drug {{ stroke:#e5e7eb; stroke-width:1.3; cursor:pointer; }}
.boxed {{ stroke:#fbbf24; stroke-width:3; }}
.node-label {{ fill:#d1d5db; font-size:11px; pointer-events:none; }}
.muted {{ opacity:.06 !important; }} .highlight {{ opacity:1 !important; stroke-width:2.5; }}
.legend {{ display:flex; gap:14px; font-size:12px; color:#cbd5e1; }}
.dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:5px; }}
</style></head><body>
<header><h1>{html.escape(title)}</h1>
<div class="sub" id="summary"></div>
<div class="controls">
<label><input id="show-serious" type="checkbox" checked> Serious patients</label>
<label><input id="show-nonserious" type="checkbox" checked> Non-serious patients</label>
<label>Edge opacity <input id="opacity" type="range" min="0.02" max="0.8" step="0.02" value="0.18"></label>
<div class="legend"><span><i class="dot" style="background:#ef4444"></i>Serious</span>
<span><i class="dot" style="background:#3b82f6"></i>Non-serious</span>
<span><i class="dot" style="background:#10b981"></i>Drug</span>
<span><i class="dot" style="background:#fbbf24"></i>Boxed-warning border</span></div>
</div></header><main id="canvas"></main>
<script>
const data={encoded};
const ns='http://www.w3.org/2000/svg';
const height=Math.max(780,Math.max(data.patients.length,data.drugs.length)*24+80), width=1200;
const svg=document.createElementNS(ns,'svg'); svg.setAttribute('viewBox',`0 0 ${{width}} ${{height}}`);
svg.setAttribute('width','100%'); svg.setAttribute('height',height); document.querySelector('#canvas').append(svg);
const serious=data.patients.filter(x=>x.serious).length;
document.querySelector('#summary').textContent=`${{data.patients.length}} patients (${{serious}} serious), ${{data.drugs.length}} drugs, ${{data.edges.length}} exposures · Hover a node to isolate its neighborhood`;
const pos={{}};
const patients=[...data.patients].sort((a,b)=>b.serious-a.serious||a.primaryid.localeCompare(b.primaryid));
patients.forEach((n,i)=>pos[n.id]=[230,55+i*(height-110)/Math.max(1,patients.length-1)]);
data.drugs.forEach((n,i)=>pos[n.id]=[940,55+i*(height-110)/Math.max(1,data.drugs.length-1)]);
const adjacency={{}}; [...data.patients,...data.drugs].forEach(n=>adjacency[n.id]=new Set());
data.edges.forEach(e=>{{adjacency[e.source].add(e.target);adjacency[e.target].add(e.source);}});
const edgeEls=[];
data.edges.forEach((e,i)=>{{const line=document.createElementNS(ns,'line');line.classList.add('edge');line.dataset.source=e.source;line.dataset.target=e.target;line.setAttribute('x1',pos[e.source][0]);line.setAttribute('y1',pos[e.source][1]);line.setAttribute('x2',pos[e.target][0]);line.setAttribute('y2',pos[e.target][1]);line.style.opacity=.18;svg.append(line);edgeEls.push(line);}});
const nodeEls={{}};
function addNode(n,type){{const [x,y]=pos[n.id],g=document.createElementNS(ns,'g'),c=document.createElementNS(ns,'circle'),t=document.createElementNS(ns,'title');g.dataset.id=n.id;g.dataset.type=type;c.setAttribute('cx',x);c.setAttribute('cy',y);c.setAttribute('r',type==='patient'?7:Math.min(16,7+Math.sqrt(n.degree||1)*1.5));c.classList.add(type);if(type==='patient')c.setAttribute('fill',n.serious?'#ef4444':'#3b82f6');else{{c.setAttribute('fill','#10b981');if(n.boxed_warning)c.classList.add('boxed');}}t.textContent=type==='patient'?`Report ${{n.primaryid}}\nSerious: ${{Boolean(n.serious)}}`:`${{n.label}}\nATC: ${{n.atc}}\nDegree: ${{n.degree}}\nBoxed warning: ${{n.boxed_warning}}\nROR z-score: ${{n.ror_z??'n/a'}}`;c.append(t);g.append(c);const label=document.createElementNS(ns,'text');label.classList.add('node-label');label.setAttribute('x',type==='patient'?x-12:x+18);label.setAttribute('y',y+4);label.setAttribute('text-anchor',type==='patient'?'end':'start');label.textContent=type==='patient'?n.primaryid:n.label.slice(0,30);g.append(label);g.addEventListener('mouseenter',()=>focus(n.id));g.addEventListener('mouseleave',clearFocus);svg.append(g);nodeEls[n.id]=g;}}
patients.forEach(n=>addNode(n,'patient'));data.drugs.forEach(n=>addNode(n,'drug'));
function focus(id){{Object.values(nodeEls).forEach(x=>x.classList.add('muted'));edgeEls.forEach(x=>x.classList.add('muted'));nodeEls[id].classList.remove('muted');nodeEls[id].classList.add('highlight');adjacency[id].forEach(other=>{{nodeEls[other].classList.remove('muted');nodeEls[other].classList.add('highlight');}});edgeEls.forEach(e=>{{if(e.dataset.source===id||e.dataset.target===id){{e.classList.remove('muted');e.classList.add('highlight');}}}});}}
function clearFocus(){{Object.values(nodeEls).forEach(x=>x.classList.remove('muted','highlight'));edgeEls.forEach(x=>x.classList.remove('muted','highlight'));applyFilters();}}
function applyFilters(){{const ss=document.querySelector('#show-serious').checked,sn=document.querySelector('#show-nonserious').checked;patients.forEach(n=>{{const visible=n.serious?ss:sn;nodeEls[n.id].style.display=visible?'':'none';}});edgeEls.forEach(e=>{{const p=patients.find(n=>n.id===e.dataset.source);e.style.display=(p.serious?ss:sn)?'':'none';e.style.opacity=document.querySelector('#opacity').value;}});}}
document.querySelectorAll('input').forEach(x=>x.addEventListener('input',applyFilters));applyFilters();
</script></body></html>"""


def _render_3d_html(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = f"TekaRx {payload['split'].title()} Patient–Drug Graph · 3D"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme:light; font-family:"Segoe UI Variable","Aptos","Segoe UI",Arial,sans-serif; }}
* {{ box-sizing:border-box; }} body {{ margin:0; overflow:hidden; background:#fff; color:#17324d;
  font-variant-numeric:tabular-nums; }}
#scene {{ position:fixed; inset:0; width:100%; height:100%; cursor:grab; }}
#scene.dragging {{ cursor:grabbing; }}
.panel {{ position:fixed; z-index:4; background:rgba(255,255,255,.96); backdrop-filter:blur(10px);
  border:1px solid #dbe5ec; box-shadow:0 10px 28px rgba(30,64,90,.12); }}
header {{ top:16px; left:16px; right:16px; padding:14px 18px 13px 21px; border-radius:12px;
  border-left:4px solid #16888b; }}
h1 {{ margin:0 0 4px; color:#15344f; font-size:20px; font-weight:650; letter-spacing:-.02em; }}
.sub {{ color:#60788b; font-size:12px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin-top:10px; }}
label,button,select {{ color:#29475f; font-size:12px; font-weight:500; }}
button,select {{ background:#f7fafb; border:1px solid #b9cbd6; padding:5px 10px; border-radius:6px;
  cursor:pointer; }} button:hover,select:hover {{ border-color:#16888b; background:#eef8f8; }}
input {{ accent-color:#16888b; }}
.legend {{ right:18px; bottom:18px; padding:10px 13px; border-radius:10px; font-size:12px;
  color:#29475f; }}
.legend div {{ margin:4px 0; }} .dot {{ width:10px; height:10px; border-radius:50%;
  display:inline-block; margin-right:7px; }}
#tooltip {{ display:none; position:fixed; z-index:6; pointer-events:none; max-width:300px;
  padding:9px 11px; border-radius:7px; color:#17324d; background:rgba(255,255,255,.98);
  border:1px solid #b9cbd6; box-shadow:0 8px 24px rgba(30,64,90,.18); font-size:12px;
  line-height:1.45; white-space:pre-line; }}
.hint {{ color:#60788b; }} input[type=range] {{ vertical-align:middle; }}
</style></head><body>
<canvas id="scene"></canvas>
<header class="panel"><h1>{html.escape(title)}</h1><div class="sub" id="summary"></div>
<div class="controls">
<span class="hint">Drag to rotate · Wheel to zoom · Hover to isolate a neighborhood</span>
<label><input id="show-serious" type="checkbox" checked> Serious</label>
<label><input id="show-nonserious" type="checkbox" checked> Non-serious</label>
<label><input id="autorotate" type="checkbox"> Auto-rotate</label>
<label>Labels <select id="labels"><option value="top" selected>Top drugs</option>
<option value="all">All nodes</option><option value="none">None</option></select></label>
<label>Edges <input id="opacity" type="range" min="0.03" max="0.7" step="0.02" value="0.18"></label>
<button id="reset">Reset view</button>
</div></header>
<aside class="panel legend"><div><i class="dot" style="background:#c54f55"></i>Serious patient</div>
<div><i class="dot" style="background:#397aa8"></i>Non-serious patient</div>
<div><i class="dot" style="background:#16888b"></i>Drug (ATC-colored)</div>
<div><i class="dot" style="background:#d99a24"></i>Boxed-warning ring</div></aside>
<div id="tooltip"></div>
<script>
const data={encoded};
const canvas=document.querySelector('#scene'),ctx=canvas.getContext('2d'),tip=document.querySelector('#tooltip');
const seriousCount=data.patients.filter(n=>n.serious).length;
document.querySelector('#summary').textContent=`${{data.patients.length}} patients (${{seriousCount}} serious) · ${{data.drugs.length}} drugs · ${{data.edges.length}} exposures`;
const palette=['#16888b','#397aa8','#d58735','#7472b2','#b85c70','#719a45','#b28a24','#368e83','#7668a8','#aa628d','#3b88ad','#c17c32','#328f72','#5f78ad','#b96872','#3c958c','#829b45','#4a88a8','#8a6dab','#ad8b32','#4b947d','#5c83ad','#a76c91','#59945d','#727daf','#b87945'];
const nodes=[],byId=new Map(),adj=new Map();
function cluster(source,type,cx,spread){{source.forEach((raw,i)=>{{const count=Math.max(1,source.length),u=(i+.5)/count,z=1-2*u,r=Math.sqrt(Math.max(0,1-z*z)),theta=i*2.3999632297;const node={{...raw,type,x:cx+Math.cos(theta)*r*spread*.45,y:Math.sin(theta)*r*spread,z:z*spread}};nodes.push(node);byId.set(node.id,node);adj.set(node.id,new Set());}});}}
cluster(data.patients,'patient',-280,330);cluster(data.drugs,'drug',280,270);
const topDrugLabels=new Set([...data.drugs].sort((a,b)=>b.degree-a.degree||a.label.localeCompare(b.label)).slice(0,15).map(n=>n.id));
data.edges.forEach(e=>{{adj.get(e.source).add(e.target);adj.get(e.target).add(e.source);}});
let rx=-.18,ry=.52,zoom=820,drag=false,lastX=0,lastY=0,hovered=null,projected=[];
function resize(){{const dpr=Math.min(window.devicePixelRatio||1,2);canvas.width=innerWidth*dpr;canvas.height=innerHeight*dpr;canvas.style.width=innerWidth+'px';canvas.style.height=innerHeight+'px';ctx.setTransform(dpr,0,0,dpr,0,0);draw();}}
function project(n){{const cy=Math.cos(ry),sy=Math.sin(ry),cx=Math.cos(rx),sx=Math.sin(rx);const x1=n.x*cy+n.z*sy,z1=-n.x*sy+n.z*cy,y2=n.y*cx-z1*sx,z2=n.y*sx+z1*cx,scale=zoom/(1050-z2);return {{node:n,x:innerWidth/2+x1*scale,y:innerHeight/2+45+y2*scale,z:z2,scale}};}}
function visible(n){{return n.type==='drug'||(n.serious?document.querySelector('#show-serious').checked:document.querySelector('#show-nonserious').checked);}}
function connected(id){{return !hovered||id===hovered||adj.get(hovered)?.has(id);}}
function nodeColor(n){{if(n.type==='patient')return n.serious?'#c54f55':'#397aa8';if(n.atc&&n.atc!=='?')return palette[(n.atc.charCodeAt(0)-65)%palette.length];return '#16888b';}}
function nodeLabel(n){{return n.type==='patient'?`Report ${{n.primaryid}}`:n.label;}}
function shouldLabel(n){{const mode=document.querySelector('#labels').value;if(mode==='none')return false;if(hovered)return n.id===hovered||adj.get(hovered)?.has(n.id);return mode==='all'||(n.type==='drug'&&topDrugLabels.has(n.id));}}
function draw(){{ctx.clearRect(0,0,innerWidth,innerHeight);ctx.fillStyle='#ffffff';ctx.fillRect(0,0,innerWidth,innerHeight);projected=nodes.filter(visible).map(project);const pmap=new Map(projected.map(p=>[p.node.id,p]));const edgeOpacity=Number(document.querySelector('#opacity').value);ctx.lineWidth=1;ctx.shadowColor='transparent';data.edges.forEach(e=>{{const a=pmap.get(e.source),b=pmap.get(e.target);if(!a||!b)return;const active=!hovered||e.source===hovered||e.target===hovered;ctx.globalAlpha=active?edgeOpacity:.025;ctx.strokeStyle=active?'#879eae':'#c9d4dc';ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}});projected.sort((a,b)=>a.z-b.z).forEach(p=>{{const n=p.node,active=connected(n.id);ctx.globalAlpha=active?Math.min(1,.58+p.scale*.5):.1;const base=n.type==='patient'?6:Math.min(15,7+Math.sqrt(n.degree||1)*1.35),radius=Math.max(3,base*p.scale);ctx.save();ctx.shadowColor='rgba(27,62,85,.24)';ctx.shadowBlur=Math.max(5,9*p.scale);ctx.shadowOffsetX=1.5;ctx.shadowOffsetY=Math.max(2,4*p.scale);ctx.beginPath();ctx.arc(p.x,p.y,radius,0,Math.PI*2);ctx.fillStyle=nodeColor(n);ctx.fill();ctx.restore();ctx.beginPath();ctx.arc(p.x,p.y,radius,0,Math.PI*2);if(n.boxed_warning){{ctx.strokeStyle='#d99a24';ctx.lineWidth=Math.max(2,3*p.scale);ctx.stroke();}}else{{ctx.strokeStyle='rgba(255,255,255,.92)';ctx.lineWidth=1.2;ctx.stroke();}}p.radius=radius;}});ctx.globalAlpha=1;ctx.shadowColor='transparent';projected.filter(p=>shouldLabel(p.node)).sort((a,b)=>a.z-b.z).forEach(p=>{{const full=nodeLabel(p.node),label=full.length>28?full.slice(0,27)+'…':full,x=p.x+p.radius+5,y=p.y-3;ctx.font=`${{hovered&&connected(p.node.id)?'650':'550'}} 11px "Segoe UI Variable","Aptos","Segoe UI",Arial,sans-serif`;ctx.lineJoin='round';ctx.lineWidth=4;ctx.strokeStyle='rgba(255,255,255,.96)';ctx.strokeText(label,x,y);ctx.fillStyle=p.node.type==='drug'?'#17324d':'#3c566b';ctx.fillText(label,x,y);}});}}
function nearest(x,y){{let best=null,dist=Infinity;for(const p of projected){{const d=Math.hypot(p.x-x,p.y-y);if(d<Math.max(11,p.radius+5)&&d<dist){{best=p;dist=d;}}}}return best;}}
function showTip(p,event){{if(!p){{tip.style.display='none';return;}}const n=p.node;tip.textContent=n.type==='patient'?`Report ${{n.primaryid}}\nSerious: ${{Boolean(n.serious)}}\nConnected drugs: ${{adj.get(n.id).size}}`:`${{n.label}}\nATC: ${{n.atc}}\nSample degree: ${{n.degree}}\nBoxed warning: ${{n.boxed_warning}}\nROR z-score: ${{n.ror_z??'n/a'}}`;tip.style.display='block';tip.style.left=Math.min(innerWidth-310,event.clientX+14)+'px';tip.style.top=Math.min(innerHeight-130,event.clientY+14)+'px';}}
canvas.addEventListener('pointerdown',e=>{{drag=true;lastX=e.clientX;lastY=e.clientY;canvas.classList.add('dragging');canvas.setPointerCapture(e.pointerId);}});
canvas.addEventListener('pointermove',e=>{{if(drag){{ry+=(e.clientX-lastX)*.008;rx+=(e.clientY-lastY)*.008;lastX=e.clientX;lastY=e.clientY;hovered=null;showTip(null,e);draw();}}else{{const p=nearest(e.clientX,e.clientY),id=p?.node.id??null;if(id!==hovered){{hovered=id;draw();}}showTip(p,e);}}}});
canvas.addEventListener('pointerup',()=>{{drag=false;canvas.classList.remove('dragging');}});canvas.addEventListener('pointerleave',()=>{{drag=false;hovered=null;canvas.classList.remove('dragging');tip.style.display='none';draw();}});
canvas.addEventListener('wheel',e=>{{e.preventDefault();zoom=Math.max(300,Math.min(1700,zoom*Math.exp(-e.deltaY*.001)));draw();}},{{passive:false}});
document.querySelector('#reset').addEventListener('click',()=>{{rx=-.18;ry=.52;zoom=820;draw();}});document.querySelectorAll('input,select').forEach(x=>x.addEventListener('input',draw));
let previous=performance.now();function animate(now){{if(document.querySelector('#autorotate').checked){{ry+=(now-previous)*.00016;draw();}}previous=now;requestAnimationFrame(animate);}}window.addEventListener('resize',resize);resize();requestAnimationFrame(animate);
</script></body></html>"""
