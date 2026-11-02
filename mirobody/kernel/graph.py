"""The patient graph: one person's record drawn as typed nodes and explained edges.

A derived read model (ADR-003). Nothing here is stored: `build` takes plain
rows the SQL layer has read for ONE user and returns the same graph for the
same rows, in any input order. Pure; stdlib only; no clock, no I/O.

    rows ── build ──▶ Graph ──┬─ local(node, depth)  a node's neighbourhood
                              └─ overview()          concepts grouped by category

Three rules decide what an edge is:

* An edge exists only where a stored column says so, and it names that column
  in `basis`. No edge is inferred from a name, a text match or two readings
  being close in time, so `EdgeType` has no causal or proximity member and
  `Edge` refuses a basis its type is not backed by (`EDGE_BASIS`).
* The vocabulary is closed. A new type needs a row in the vault's
  "Patient Graph Model" table, a member here and a test.
* A response is capped, and a capped response says what was cut
  (`Graph.truncated`, `Graph.cut`), in counts only: a notice is logged and
  shown, so it never carries a name or a value.

Device readings are never nodes. A wearable writes thousands of rows a day;
they collapse into their Concept (`count`, `first`, `last`) and that Concept
points at the DeviceStream that recorded it.
"""

from __future__ import annotations

import hashlib
import hmac
from collections import deque
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum

GRAPH_SCHEMA_VERSION = 1

#: Per-response caps (FEAT-014 FR-6), and how far `local` may look.
MAX_NODES = 500
MAX_EDGES = 2000
MAX_DEPTH = 3

#: `th_observation.modality` of a reading that comes off a sensor.
DEVICE_SENSED = "device-sensed"
#: `th_observation.kind` of a row that groups other rows (a lab panel).
ORGANIZER = "organizer"
STATUS_FINAL = "final"
UNCATEGORIZED = "uncategorized"


class NodeType(StrEnum):
    PATIENT = "Patient"
    OBSERVATION = "Observation"
    PANEL = "Panel"
    CONCEPT = "Concept"
    EXTRACTION = "Extraction"
    SOURCE_DOCUMENT = "SourceDocument"
    DEVICE_STREAM = "DeviceStream"
    MEDICATION_PLAN = "MedicationPlan"
    MEDICATION_COURSE = "MedicationCourse"


class EdgeType(StrEnum):
    HAS_RECORD = "HAS_RECORD"
    MEASURES = "MEASURES"
    MEMBER_OF = "MEMBER_OF"
    EXTRACTED_FROM = "EXTRACTED_FROM"
    READ_FROM = "READ_FROM"
    RECORDED_BY = "RECORDED_BY"
    AMENDS = "AMENDS"
    DERIVED_FROM = "DERIVED_FROM"
    COURSE_OF = "COURSE_OF"


#: The column each edge type may name as its `basis`. Rule 5 of the model: an
#: edge says why two nodes are connected, and "because" is always a column.
EDGE_BASIS: Mapping[EdgeType, frozenset[str]] = {
    EdgeType.HAS_RECORD: frozenset({"th_observation.user_id", "th_files.user_id", "th_medication_plan.user_id"}),
    EdgeType.MEASURES: frozenset({"th_coding_current.series_id"}),
    EdgeType.MEMBER_OF: frozenset({"th_observation.member_of"}),
    EdgeType.EXTRACTED_FROM: frozenset({"th_observation.extraction_id"}),
    EdgeType.READ_FROM: frozenset({"th_extraction.source_ref"}),
    EdgeType.RECORDED_BY: frozenset({"th_observation.stream_key"}),
    EdgeType.AMENDS: frozenset({"th_observation.amends"}),
    EdgeType.DERIVED_FROM: frozenset({"th_observation.derived_from"}),
    EdgeType.COURSE_OF: frozenset({"th_medication_course.plan_id"}),
}

#: The id prefix of each node type. A Concept keeps its series id unprefixed:
#: that id is already a code, not a row number.
_PREFIX: Mapping[NodeType, str] = {
    NodeType.PATIENT: "pt",
    NodeType.OBSERVATION: "obs",
    NodeType.PANEL: "pnl",
    NodeType.EXTRACTION: "ext",
    NodeType.SOURCE_DOCUMENT: "doc",
    NodeType.MEDICATION_PLAN: "med",
    NodeType.MEDICATION_COURSE: "crs",
    NodeType.DEVICE_STREAM: "dev",
}

# --- the graph -----------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    id: str
    type: NodeType
    label: str
    props: Mapping[str, object] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        # A string is accepted only if it names a member: the vocabulary stays
        # closed at the constructor, whatever a loader passes.
        object.__setattr__(self, "type", NodeType(self.type))


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    type: EdgeType
    basis: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", EdgeType(self.type))
        if self.basis not in EDGE_BASIS[self.type]:
            raise ValueError(f"{self.type.value} is not backed by column {self.basis!r}")


@dataclass(frozen=True)
class Graph:
    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()
    graph_schema_version: int = GRAPH_SCHEMA_VERSION
    # `truncated` and `cut` mean what they mean on `tools.Meta`: the answer is
    # partial, and `cut` says in plain words what is missing and the call that
    # shows more.
    truncated: bool = False
    cut: str = ""
    omitted_nodes: int = 0
    omitted_edges: int = 0
    # Edges not drawn because one end is not in the graph (a row outside the
    # loaded window, or a device reading that is a Concept now). Counted so a
    # loader that filters too hard can see it.
    dangling: int = 0


@dataclass(frozen=True)
class Group:
    key: str
    label: str
    concepts: tuple[str, ...]  # Concept node ids, most readings first
    readings: int


@dataclass(frozen=True)
class Overview:
    groups: tuple[Group, ...]
    graph: Graph


# --- input rows ----------------------------------------------------------------------
# Plain values, one dataclass per table the SQL layer reads. Field names are the
# columns'. Free text that is encrypted at rest arrives already decrypted for the
# caller who may read it; this module never asks for the key.


@dataclass(frozen=True)
class ObservationRow:
    """A `v_observation`-shaped row, read WITHOUT the amended/status filter when
    history is wanted: `build` decides which rows are heads."""

    id: int
    kind: str
    modality: str
    name_text: str
    observed_start: datetime
    local_date: date
    series_id: str
    status: str = STATUS_FINAL
    source_kind: str = ""
    vendor: str = ""
    vendor_field: str = ""
    stream_key: str = ""
    source_class: str = ""
    extraction_id: int | None = None
    row_ix: int | None = None
    member_of: int | None = None
    amends: int | None = None
    derived_from: tuple[int, ...] = ()
    panel_text: str = ""
    grain: str = ""
    value_text: str = ""
    value_num: float | None = None
    unit_text: str = ""
    unit_ucum: str = ""
    ref_text: str = ""
    flag_text: str = ""
    # th_coding_current / th_concept
    outcome: str = ""
    code_system: str = ""
    code: str = ""
    display: str = ""
    category: str = ""  # the catalogue category or LOINC class the loader resolved
    elected: bool = False


@dataclass(frozen=True)
class ExtractionRow:
    id: int
    source_kind: str
    source_ref: str
    extractor: str
    status: str
    report_date: date | None = None
    date_source: str = ""


@dataclass(frozen=True)
class FileRow:
    """Metadata only: no original name, no text, no bytes."""

    file_key: str
    file_type: str = ""
    scene: str = ""
    created_at: datetime | None = None


@dataclass(frozen=True)
class PlanRow:
    plan_id: str
    concept_key: str
    status: str
    start_date: date
    end_date: date | None = None
    confirmed: bool = True
    source: str = ""
    label: str = ""  # the drug name as the person wrote it, when the caller may read it


@dataclass(frozen=True)
class CourseRow:
    course_id: str
    plan_id: str
    start_date: date
    end_date: date | None = None
    closed_by: str = ""


# --- ids -----------------------------------------------------------------------------


def _node_id(node_type: NodeType, internal_id: object) -> str:
    return f"{_PREFIX[node_type]}:{internal_id}"


def opaque_id(key: bytes, node_type: NodeType, internal_id: object) -> str:
    """The id a client sees: `<prefix>_<16 hex>`, the first 64 bits of an
    HMAC-SHA256 over the type and the row's own id. Stable for one key, unlinkable
    across keys, and a client cannot compute another person's ids without it.
    A Concept keeps its series id: that is a LOINC-derived code, shared by every
    person, and says nothing about whose record it is.

    Where the key lives and how it is scoped to a person is the caller's.
    """
    node_type = NodeType(node_type)
    if node_type is NodeType.CONCEPT:
        return str(internal_id)
    if not key:
        raise ValueError("opaque ids need a non-empty key")
    mac = hmac.new(key, f"{node_type.value}:{internal_id}".encode(), hashlib.sha256).digest()
    return f"{_PREFIX[node_type]}_{mac[:8].hex()}"


def opaque_graph(graph: Graph, key: bytes) -> tuple[Graph, dict[str, str]]:
    """The graph with every node id replaced by its opaque id, and the map back
    (opaque -> internal) the server keeps to resolve a request."""
    mapping: dict[str, str] = {}
    forward: dict[str, str] = {}
    for n in graph.nodes:
        raw = n.id if n.type is NodeType.CONCEPT else n.id.split(":", 1)[1]
        o = opaque_id(key, n.type, raw)
        forward[n.id] = o
        mapping[o] = n.id
    nodes = tuple(sorted((Node(forward[n.id], n.type, n.label, n.props) for n in graph.nodes), key=_node_key))
    edges = tuple(
        sorted(
            (Edge(forward[e.source], forward[e.target], e.type, e.basis) for e in graph.edges),
            key=_edge_key,
        )
    )
    return _replace(graph, nodes=nodes, edges=edges), mapping


# --- build ---------------------------------------------------------------------------


def _node_key(n: Node) -> tuple[str, str]:
    return (n.type.value, n.id)


def _edge_key(e: Edge) -> tuple[str, str, str, str]:
    return (e.type.value, e.source, e.target, e.basis)


def _replace(g: Graph, **changes: object) -> Graph:
    data = {f: getattr(g, f) for f in g.__dataclass_fields__}
    data.update(changes)
    return Graph(**data)  # type: ignore[arg-type]


def _iso(d: date | datetime | None) -> str | None:
    return d.isoformat() if d is not None else None


def build(
    observations: Iterable[ObservationRow] = (),
    extractions: Iterable[ExtractionRow] = (),
    files: Iterable[FileRow] = (),
    plans: Iterable[PlanRow] = (),
    courses: Iterable[CourseRow] = (),
    *,
    patient_id: str = "",
    patient_label: str = "",
    include_history: bool = False,
) -> Graph:
    """Rows of one person in, graph out. Same rows in any order give the same graph.

    Only current heads are drawn unless `include_history`: a row is a head when
    it is `final` and no other row amends it, which is `v_observation`'s own
    rule. With history, the superseded and entered-in-error rows come back and
    each amending row gets an AMENDS edge to the row it replaced.
    """
    obs_all = sorted(observations, key=lambda o: o.id)
    amended = {o.amends for o in obs_all if o.amends is not None}
    heads = [o for o in obs_all if o.status == STATUS_FINAL and o.id not in amended]
    drawn = obs_all if include_history else heads
    head_ids = {o.id for o in heads}

    nodes: dict[str, Node] = {}
    edges: set[Edge] = set()
    dangling = 0

    def add_edge(source: str, target: str, etype: EdgeType, basis: str) -> None:
        nonlocal dangling
        if source in nodes and target in nodes:
            edges.add(Edge(source, target, etype, basis))
        else:
            dangling += 1

    # Nodes first, so an edge's two ends can be checked whatever the row order.
    ext_rows = sorted(extractions, key=lambda x: x.id)
    for x in ext_rows:
        nodes[_node_id(NodeType.EXTRACTION, x.id)] = Node(
            _node_id(NodeType.EXTRACTION, x.id),
            NodeType.EXTRACTION,
            x.extractor,
            {
                "source_kind": x.source_kind,
                "extractor": x.extractor,
                "status": x.status,
                "report_date": _iso(x.report_date),
                "date_source": x.date_source,
            },
        )
    file_rows = sorted(files, key=lambda f: f.file_key)
    file_ids: dict[str, str] = {}
    for f in file_rows:
        nid = _node_id(NodeType.SOURCE_DOCUMENT, f.file_key)
        file_ids[f.file_key] = nid
        nodes[nid] = Node(
            nid,
            NodeType.SOURCE_DOCUMENT,
            f.scene or "document",  # never the file's name: it is a person's text
            {"file_type": f.file_type, "scene": f.scene, "created_at": _iso(f.created_at)},
        )

    plan_rows = [p for p in sorted(plans, key=lambda p: p.plan_id) if include_history or p.status != "entered_in_error"]
    for p in plan_rows:
        nid = _node_id(NodeType.MEDICATION_PLAN, p.plan_id)
        nodes[nid] = Node(
            nid,
            NodeType.MEDICATION_PLAN,
            p.label or p.concept_key,
            {
                "concept_key": p.concept_key,
                "status": p.status,
                "start_date": _iso(p.start_date),
                "end_date": _iso(p.end_date),
                "confirmed": p.confirmed,
                "source": p.source,
            },
        )
    course_rows = [
        c
        for c in sorted(courses, key=lambda c: c.course_id)
        if include_history or c.closed_by != "entered_in_error"
    ]
    for c in course_rows:
        nid = _node_id(NodeType.MEDICATION_COURSE, c.course_id)
        nodes[nid] = Node(
            nid,
            NodeType.MEDICATION_COURSE,
            f"{c.start_date.isoformat()}",
            {"start_date": _iso(c.start_date), "end_date": _iso(c.end_date), "closed_by": c.closed_by},
        )

    # Observations: a panel is an organizer row, a sensor reading is not a node.
    series: dict[str, list[ObservationRow]] = {}
    streams: dict[str, ObservationRow] = {}
    collapsed_ids: set[int] = set()
    node_of: dict[int, str] = {}
    for o in drawn:
        if o.kind != ORGANIZER and o.series_id and o.id in head_ids:
            series.setdefault(o.series_id, []).append(o)
        if o.modality == DEVICE_SENSED and o.kind != ORGANIZER:
            collapsed_ids.add(o.id)
            if o.vendor and o.id in head_ids:
                streams.setdefault(o.stream_key or f"{o.series_id}|{o.vendor}", o)
            continue
        ntype = NodeType.PANEL if o.kind == ORGANIZER else NodeType.OBSERVATION
        nid = _node_id(ntype, o.id)
        node_of[o.id] = nid
        if ntype is NodeType.PANEL:
            nodes[nid] = Node(nid, ntype, o.panel_text or o.name_text, {"observed_start": _iso(o.observed_start)})
        else:
            nodes[nid] = Node(
                nid,
                ntype,
                o.name_text,
                {
                    "kind": o.kind,
                    "modality": o.modality,
                    "grain": o.grain,
                    "value_text": o.value_text,
                    "value_num": o.value_num,
                    "unit_text": o.unit_text,
                    "unit_ucum": o.unit_ucum,
                    "ref_text": o.ref_text,
                    "flag_text": o.flag_text,
                    "observed_start": _iso(o.observed_start),
                    "local_date": _iso(o.local_date),
                    "status": o.status,
                    "outcome": o.outcome,
                    "code_system": o.code_system,
                    "code": o.code,
                    "row_ix": o.row_ix,
                    "elected": o.elected,
                },
            )
            if o.source_kind == "device" and o.vendor:
                streams.setdefault(o.stream_key or f"{o.series_id}|{o.vendor}", o)

    # Concepts count current readings only: a superseded value is history, not a reading.
    concept_ids: dict[str, str] = {}
    for sid, rows in sorted(series.items()):
        rows.sort(key=lambda r: r.id)
        starts = [r.observed_start for r in rows]
        rep = next((r for r in rows if r.display), rows[0])
        concept_ids[sid] = sid
        nodes[sid] = Node(
            sid,
            NodeType.CONCEPT,
            rep.display or rep.name_text,
            {
                "code_system": rep.code_system,
                "code": rep.code,
                "category": next((r.category for r in rows if r.category), ""),
                "count": len(rows),
                "first": _iso(min(starts)),
                "last": _iso(max(starts)),
            },
        )
    stream_ids: dict[str, str] = {}
    for skey, o in sorted(streams.items()):
        nid = _node_id(NodeType.DEVICE_STREAM, skey)
        stream_ids[skey] = nid
        nodes[nid] = Node(
            nid,
            NodeType.DEVICE_STREAM,
            o.vendor,
            {"vendor": o.vendor, "vendor_field": o.vendor_field, "source_class": o.source_class},
        )

    # Edges.
    for x in ext_rows:
        if x.source_kind == "file" and x.source_ref:
            add_edge(
                _node_id(NodeType.EXTRACTION, x.id),
                file_ids.get(x.source_ref, ""),
                EdgeType.READ_FROM,
                "th_extraction.source_ref",
            )
    for c in course_rows:
        add_edge(
            _node_id(NodeType.MEDICATION_COURSE, c.course_id),
            _node_id(NodeType.MEDICATION_PLAN, c.plan_id),
            EdgeType.COURSE_OF,
            "th_medication_course.plan_id",
        )
    for o in drawn:
        if o.id in collapsed_ids:
            # A collapsed reading has no node of its own; its Concept carries the stream.
            continue
        src = node_of[o.id]
        if o.kind != ORGANIZER and o.series_id:
            add_edge(src, concept_ids.get(o.series_id, ""), EdgeType.MEASURES, "th_coding_current.series_id")
        if o.member_of is not None:
            add_edge(src, node_of.get(o.member_of, ""), EdgeType.MEMBER_OF, "th_observation.member_of")
        if o.extraction_id is not None:
            add_edge(
                src,
                _node_id(NodeType.EXTRACTION, o.extraction_id),
                EdgeType.EXTRACTED_FROM,
                "th_observation.extraction_id",
            )
        if o.source_kind == "device" and o.vendor:
            add_edge(
                src,
                stream_ids.get(o.stream_key or f"{o.series_id}|{o.vendor}", ""),
                EdgeType.RECORDED_BY,
                "th_observation.stream_key",
            )
        if o.amends is not None and include_history:
            add_edge(src, node_of.get(o.amends, ""), EdgeType.AMENDS, "th_observation.amends")
        for parent in o.derived_from:
            add_edge(src, node_of.get(parent, ""), EdgeType.DERIVED_FROM, "th_observation.derived_from")
    # The Concept stands in for its sensor readings, so it is the one a stream records.
    for sid, rows in series.items():
        for r in rows:
            if r.id in collapsed_ids and r.vendor:
                add_edge(
                    concept_ids[sid],
                    stream_ids.get(r.stream_key or f"{r.series_id}|{r.vendor}", ""),
                    EdgeType.RECORDED_BY,
                    "th_observation.stream_key",
                )

    if patient_id:
        pid = _node_id(NodeType.PATIENT, patient_id)
        nodes[pid] = Node(pid, NodeType.PATIENT, patient_label or "you", {})
        # Top level = nothing above it to reach it by: documents, panels, plans,
        # streams, and readings no document, panel or sensor accounts for.
        for n in sorted(nodes.values(), key=_node_key):
            basis = None
            if n.type in (NodeType.SOURCE_DOCUMENT,):
                basis = "th_files.user_id"
            elif n.type is NodeType.MEDICATION_PLAN:
                basis = "th_medication_plan.user_id"
            elif n.type in (NodeType.PANEL, NodeType.DEVICE_STREAM):
                basis = "th_observation.user_id"
            if basis:
                edges.add(Edge(pid, n.id, EdgeType.HAS_RECORD, basis))
        for o in drawn:
            if o.id in node_of and node_of[o.id].startswith("obs:") and o.extraction_id is None and o.member_of is None:
                edges.add(Edge(pid, node_of[o.id], EdgeType.HAS_RECORD, "th_observation.user_id"))

    return Graph(
        nodes=tuple(sorted(nodes.values(), key=_node_key)),
        edges=tuple(sorted(edges, key=_edge_key)),
        dangling=dangling,
    )


# --- views ---------------------------------------------------------------------------


def _cut_notice(shown_n: int, total_n: int, shown_e: int, total_e: int, next_call: str) -> str:
    return (
        f"Showing {shown_n} of {total_n} nodes and {shown_e} of {total_e} edges; "
        f"{total_n - shown_n} nodes and {total_e - shown_e} edges are not shown. {next_call}"
    )


def _finish(
    base: Graph, ordered: Sequence[Node], edges: Iterable[Edge], max_nodes: int, max_edges: int, next_call: str
) -> Graph:
    """Keep the first `max_nodes` of `ordered`, the edges between them up to
    `max_edges` (in sorted order, so the same cut every time), and say what went."""
    edge_set = set(edges)
    kept = {n.id for n in ordered[:max_nodes]}
    inside = sorted((e for e in edge_set if e.source in kept and e.target in kept), key=_edge_key)
    all_edges = len(edge_set)
    shown_e = inside[:max_edges]
    omitted_n = len(ordered) - len(kept)
    omitted_e = all_edges - len(shown_e)
    truncated = omitted_n > 0 or omitted_e > 0
    return Graph(
        nodes=tuple(sorted((n for n in ordered if n.id in kept), key=_node_key)),
        edges=tuple(shown_e),
        graph_schema_version=base.graph_schema_version,
        truncated=truncated,
        cut=_cut_notice(len(kept), len(ordered), len(shown_e), all_edges, next_call) if truncated else "",
        omitted_nodes=omitted_n,
        omitted_edges=omitted_e,
        dangling=base.dangling,
    )


def local(
    graph: Graph,
    node_id: str,
    depth: int = 1,
    *,
    max_nodes: int = MAX_NODES,
    max_edges: int = MAX_EDGES,
) -> Graph:
    """The neighbourhood of one node, `depth` (1-3) hops out over any edge in
    either direction. Raises `LookupError` for an id the graph does not hold, which
    a caller answers as 404 so a foreign id and a missing one look the same.

    The Patient hangs off everything at depth 2, so HAS_RECORD is not walked unless
    the walk starts at the Patient.
    """
    if not 1 <= depth <= MAX_DEPTH:
        raise ValueError(f"depth must be between 1 and {MAX_DEPTH}")
    by_id = {n.id: n for n in graph.nodes}
    if node_id not in by_id:
        raise LookupError("node not in graph")
    walk_has_record = by_id[node_id].type is NodeType.PATIENT
    usable = [e for e in graph.edges if walk_has_record or e.type is not EdgeType.HAS_RECORD]
    adj: dict[str, list[str]] = {}
    for e in usable:
        adj.setdefault(e.source, []).append(e.target)
        adj.setdefault(e.target, []).append(e.source)
    seen = {node_id}
    order = [node_id]
    queue: deque[tuple[str, int]] = deque([(node_id, 0)])
    while queue:
        cur, d = queue.popleft()
        if d == depth:
            continue
        for nxt in sorted(set(adj.get(cur, ())), key=lambda i: (by_id[i].type.value, i)):
            if nxt not in seen:
                seen.add(nxt)
                order.append(nxt)
                queue.append((nxt, d + 1))
    within = set(order)
    return _finish(
        graph,
        [by_id[i] for i in order],
        (e for e in usable if e.source in within and e.target in within),
        max_nodes,
        max_edges,
        "Ask for the node itself at depth 1, or for a node closer to what you want, to see the rest.",
    )


#: Types the overview draws by default: the things a record is made of, not its
#: individual readings (those are one `local` away).
OVERVIEW_TYPES = frozenset(
    {
        NodeType.PATIENT,
        NodeType.CONCEPT,
        NodeType.SOURCE_DOCUMENT,
        NodeType.DEVICE_STREAM,
        NodeType.MEDICATION_PLAN,
        NodeType.MEDICATION_COURSE,
    }
)

_OVERVIEW_RANK: Mapping[NodeType, int] = {
    NodeType.PATIENT: 0,
    NodeType.MEDICATION_PLAN: 1,
    NodeType.SOURCE_DOCUMENT: 2,
    NodeType.DEVICE_STREAM: 3,
    NodeType.CONCEPT: 4,
    NodeType.MEDICATION_COURSE: 5,
}


def overview(
    graph: Graph,
    *,
    types: Collection[NodeType] | None = None,
    max_nodes: int = MAX_NODES,
    max_edges: int = MAX_EDGES,
) -> Overview:
    """Concepts grouped by category, beside the documents, streams and
    medications they connect to. When over the cap, the busiest concepts stay."""
    wanted = frozenset(NodeType(t) for t in types) if types is not None else OVERVIEW_TYPES
    pool = [n for n in graph.nodes if n.type in wanted]

    def rank(n: Node) -> tuple[int, int, str]:
        count = n.props.get("count", 0)
        return (_OVERVIEW_RANK.get(n.type, 9), -(count if isinstance(count, int) else 0), n.id)

    ordered = sorted(pool, key=rank)
    pool_ids = {n.id for n in pool}
    shown = _finish(
        graph,
        ordered,
        [e for e in graph.edges if e.source in pool_ids and e.target in pool_ids],
        max_nodes,
        max_edges,
        "Narrow it with a node type or a date range, or open one node with local.",
    )
    by_cat: dict[str, list[Node]] = {}
    for n in shown.nodes:
        if n.type is NodeType.CONCEPT:
            by_cat.setdefault(str(n.props.get("category") or UNCATEGORIZED), []).append(n)
    groups = tuple(
        Group(
            key=cat,
            label=cat,
            concepts=tuple(c.id for c in sorted(cs, key=lambda c: (-int(c.props.get("count", 0)), c.id))),  # type: ignore[call-overload]
            readings=sum(int(c.props.get("count", 0)) for c in cs),  # type: ignore[call-overload]
        )
        for cat, cs in sorted(by_cat.items())
    )
    return Overview(groups=groups, graph=shown)


def as_dict(graph: Graph) -> dict[str, object]:
    """The JSON shape of a graph: plain types, the sorted order `build` made."""
    return {
        "graph_schema_version": graph.graph_schema_version,
        "nodes": [{"id": n.id, "type": n.type.value, "label": n.label, "props": dict(n.props)} for n in graph.nodes],
        "edges": [{"source": e.source, "target": e.target, "type": e.type.value, "basis": e.basis} for e in graph.edges],
        "truncated": graph.truncated,
        "cut": graph.cut,
    }
