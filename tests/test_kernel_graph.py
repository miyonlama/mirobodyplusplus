"""mirobody.kernel.graph on invented rows: no database, no clock.

Graph Test Plan G-01..G-07 (vault: 10 - Testing and Quality).
"""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, timedelta

import pytest

from mirobody.kernel import graph as g
from mirobody.kernel.graph import EdgeType, NodeType

T0 = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
KEY = b"k" * 32


def obs(i: int, **kw) -> g.ObservationRow:
    base = {
        "id": i,
        "kind": "measurement",
        "modality": "lab-report",
        "name_text": f"Analyte {i}",
        "observed_start": T0 + timedelta(minutes=i),
        "local_date": date(2026, 3, 1),
        "series_id": "loinc:2093-3",
        "outcome": "coded",
        "code_system": "LOINC",
        "code": "2093-3",
        "display": "Cholesterol",
        "category": "lipids",
    }
    base.update(kw)
    return g.ObservationRow(**base)


def fixture() -> dict:
    """A panel read from a document, one amended reading, a derived one, a plan with a course,
    and a heart-rate stream, so every edge type has an instance."""
    return {
        "observations": [
            obs(1, kind="organizer", series_id="", panel_text="Lipid panel", name_text="Lipid panel", extraction_id=5),
            obs(2, member_of=1, extraction_id=5, row_ix=0),
            obs(3, member_of=1, extraction_id=5, row_ix=1, series_id="loinc:2085-9", name_text="HDL", display="HDL"),
            obs(4, modality="derived", derived_from=(2, 3), series_id="loinc:9830-1", name_text="Ratio", display="Ratio"),
            obs(10, name_text="Old weight", series_id="loinc:29463-7", kind="measurement", modality="manual"),
            obs(11, name_text="Weight", series_id="loinc:29463-7", modality="manual", amends=10),
        ],
        "extractions": [g.ExtractionRow(5, "file", "files/a1", "llm:x/y@draft-v1", "ok", date(2026, 2, 20), "extracted")],
        "files": [g.FileRow("files/a1", "application/pdf", "report", T0)],
        "plans": [g.PlanRow("p1", "rxnorm:1", "active", date(2026, 1, 1), label="Metformin")],
        "courses": [g.CourseRow("c1", "p1", date(2026, 1, 1), None, "")],
        "patient_id": "7",
    }


def hr(i: int) -> g.ObservationRow:
    return obs(
        1000 + i,
        modality=g.DEVICE_SENSED,
        source_kind="device",
        vendor="acme",
        stream_key="hr|device:acme",
        source_class="measurer",
        series_id="loinc:8867-4",
        name_text="Heart rate",
        display="Heart rate",
        category="vitals",
        observed_start=T0 + timedelta(seconds=i),
    )


def edge_types(graph: g.Graph) -> set[EdgeType]:
    return {e.type for e in graph.edges}


# G-01
def test_every_edge_type_is_produced_with_its_basis():
    rows = fixture()
    rows["observations"] = rows["observations"] + [
        obs(20, modality="manual", source_kind="device", vendor="scale", stream_key="w|device:scale", series_id="loinc:1")
    ]
    graph = g.build(**rows, include_history=True)
    assert edge_types(graph) == set(EdgeType)
    for e in graph.edges:
        assert e.basis in g.EDGE_BASIS[e.type]
    by = {(e.type, e.source, e.target): e.basis for e in graph.edges}
    assert by[(EdgeType.MEASURES, "obs:2", "loinc:2093-3")] == "th_coding_current.series_id"
    assert by[(EdgeType.MEMBER_OF, "obs:2", "pnl:1")] == "th_observation.member_of"
    assert by[(EdgeType.EXTRACTED_FROM, "obs:2", "ext:5")] == "th_observation.extraction_id"
    assert by[(EdgeType.READ_FROM, "ext:5", "doc:files/a1")] == "th_extraction.source_ref"
    assert by[(EdgeType.AMENDS, "obs:11", "obs:10")] == "th_observation.amends"
    assert by[(EdgeType.DERIVED_FROM, "obs:4", "obs:2")] == "th_observation.derived_from"
    assert by[(EdgeType.COURSE_OF, "crs:c1", "med:p1")] == "th_medication_course.plan_id"
    assert by[(EdgeType.HAS_RECORD, "pt:7", "doc:files/a1")] == "th_files.user_id"


def test_recorded_by_for_collapsed_readings_comes_from_the_concept():
    graph = g.build([hr(0)])
    assert [(e.source, e.target, e.basis) for e in graph.edges if e.type is EdgeType.RECORDED_BY] == [
        ("loinc:8867-4", "dev:hr|device:acme", "th_observation.stream_key")
    ]


def test_document_node_carries_no_file_name_or_text():
    graph = g.build(files=[g.FileRow("files/a1", "application/pdf", "report", T0)])
    (doc,) = graph.nodes
    assert doc.label == "report"
    assert set(doc.props) == {"file_type", "scene", "created_at"}


def test_edge_to_a_node_that_is_not_loaded_is_counted_not_drawn():
    graph = g.build([obs(2, member_of=1, extraction_id=5)])
    assert edge_types(graph) == {EdgeType.MEASURES}
    assert graph.dangling == 2


# G-02 / G-07
def test_vocabulary_is_closed():
    with pytest.raises(ValueError):
        EdgeType("CAUSED_BY")
    with pytest.raises(ValueError):
        NodeType("Encounter")
    with pytest.raises(ValueError):
        g.Edge("a", "b", "TREATS", "th_observation.member_of")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        g.Node("a", "Provider", "x")  # type: ignore[arg-type]
    assert {e.value for e in EdgeType} == {
        "HAS_RECORD", "MEASURES", "MEMBER_OF", "EXTRACTED_FROM", "READ_FROM",
        "RECORDED_BY", "AMENDS", "DERIVED_FROM", "COURSE_OF",
    }  # fmt: skip


def test_an_edge_must_name_the_column_that_backs_it():
    with pytest.raises(ValueError):
        g.Edge("obs:1", "pnl:2", EdgeType.MEMBER_OF, "same day")
    with pytest.raises(ValueError):
        g.Edge("obs:1", "pnl:2", EdgeType.MEMBER_OF, "th_coding_current.series_id")


def test_no_causal_or_proximity_edge_exists():
    banned = ("CAUS", "TREAT", "WORSEN", "IMPROVE", "NEAR", "PROXIM", "SIMILAR", "RELATED", "CORREL", "SAME_DAY", "BEFORE", "AFTER")
    for name in (e.name for e in EdgeType):
        assert not any(b in name for b in banned), name
    assert set(g.EDGE_BASIS) == set(EdgeType)
    # Two readings on one day with one name get no edge between them.
    graph = g.build([obs(1), obs(2)])
    assert not [e for e in graph.edges if e.type not in (EdgeType.MEASURES,)]


# G-03
def test_ten_thousand_device_readings_are_one_concept():
    rows = [hr(i) for i in range(10_000)]
    graph = g.build(rows, patient_id="7")
    assert [n.type for n in graph.nodes if n.type is NodeType.OBSERVATION] == []
    (concept,) = [n for n in graph.nodes if n.type is NodeType.CONCEPT]
    assert concept.props["count"] == 10_000
    assert concept.props["first"] == T0.isoformat()
    assert concept.props["last"] == (T0 + timedelta(seconds=9_999)).isoformat()
    assert len([n for n in graph.nodes if n.type is NodeType.DEVICE_STREAM]) == 1
    assert len(graph.edges) == 2  # RECORDED_BY, HAS_RECORD


# G-04
def test_amended_rows_are_head_only_unless_history_is_asked_for():
    rows = fixture()
    now = g.build(**rows)
    ids = {n.id for n in now.nodes}
    assert "obs:11" in ids and "obs:10" not in ids
    assert EdgeType.AMENDS not in edge_types(now)
    (weight,) = [n for n in now.nodes if n.id == "loinc:29463-7"]
    assert weight.props["count"] == 1

    past = g.build(**rows, include_history=True)
    assert {"obs:10", "obs:11"} <= {n.id for n in past.nodes}
    assert (EdgeType.AMENDS, "obs:11", "obs:10") in {(e.type, e.source, e.target) for e in past.edges}
    (weight,) = [n for n in past.nodes if n.id == "loinc:29463-7"]
    assert weight.props["count"] == 1  # a replaced value is not a reading


def test_entered_in_error_hides_both_rows_by_default():
    rows = [obs(10, modality="manual"), obs(11, modality="manual", amends=10, status="entered-in-error")]
    assert not [n for n in g.build(rows).nodes if n.type is NodeType.OBSERVATION]
    shown = g.build(rows, include_history=True)
    assert {n.id for n in shown.nodes if n.type is NodeType.OBSERVATION} == {"obs:10", "obs:11"}


# local / acceptance criteria
def test_local_depth_one_has_three_edges_and_read_from_arrives_at_depth_two():
    graph = g.build(**fixture())
    one = g.local(graph, "obs:2", 1)
    assert {EdgeType.MEASURES, EdgeType.EXTRACTED_FROM, EdgeType.MEMBER_OF} <= edge_types(one)
    assert EdgeType.READ_FROM not in edge_types(one)
    two = g.local(graph, "obs:2", 2)
    assert EdgeType.READ_FROM in edge_types(two)
    assert all(e.basis for e in two.edges)


def test_local_rejects_bad_depth_and_unknown_node():
    graph = g.build(**fixture())
    for depth in (0, 4):
        with pytest.raises(ValueError):
            g.local(graph, "obs:2", depth)
    with pytest.raises(LookupError):
        g.local(graph, "obs:999", 1)


def test_local_does_not_walk_through_the_patient():
    graph = g.build(**fixture())
    sub = g.local(graph, "obs:2", 3)
    assert "pt:7" not in {n.id for n in sub.nodes}
    assert EdgeType.HAS_RECORD not in edge_types(sub)


# G-05
def test_local_caps_nodes_and_says_what_was_cut():
    rows = [obs(i, extraction_id=1, modality="manual", series_id=f"s{i}") for i in range(1, 801)]
    graph = g.build(rows, [g.ExtractionRow(1, "manual", "", "manual", "ok")])
    sub = g.local(graph, "ext:1", 1)
    assert len(sub.nodes) == g.MAX_NODES
    assert sub.truncated and sub.omitted_nodes == 801 - g.MAX_NODES
    assert "Showing 500 of 801 nodes" in sub.cut
    assert "ext:1" in {n.id for n in sub.nodes}
    # The notice carries counts, never a name or a value.
    assert "Analyte" not in sub.cut


def test_edge_cap_truncates_in_sorted_order():
    rows = [obs(i, extraction_id=1, modality="manual", series_id="s") for i in range(1, 51)]
    graph = g.build(rows, [g.ExtractionRow(1, "manual", "", "manual", "ok")])
    sub = g.local(graph, "ext:1", 1, max_edges=10)
    assert len(sub.edges) == 10 and sub.truncated and sub.omitted_edges == 40
    assert sub.edges == tuple(sorted(sub.edges, key=g._edge_key))


def test_a_complete_answer_is_not_marked_truncated():
    sub = g.local(g.build(**fixture()), "obs:2", 2)
    assert not sub.truncated and sub.cut == "" and sub.omitted_nodes == 0


def test_overview_groups_concepts_by_category_and_collapses_readings():
    rows = fixture()
    rows["observations"] = rows["observations"] + [hr(i) for i in range(50)]
    ov = g.overview(g.build(**rows))
    groups = {grp.key: grp for grp in ov.groups}
    assert groups["vitals"].concepts == ("loinc:8867-4",) and groups["vitals"].readings == 50
    assert groups["lipids"].readings == 4  # heads 2, 3, 4 and 11; the replaced 10 is not a reading
    assert not [n for n in ov.graph.nodes if n.type in (NodeType.OBSERVATION, NodeType.EXTRACTION, NodeType.PANEL)]
    assert not ov.graph.truncated


def test_overview_cap_keeps_the_busiest_concepts():
    rows = [obs(i, modality="manual", series_id=f"s{i % 40}") for i in range(1, 401)]
    rows += [obs(1000 + i, modality="manual", series_id="busy") for i in range(100)]
    ov = g.overview(g.build(rows), max_nodes=5)
    assert ov.graph.truncated and len(ov.graph.nodes) == 5
    assert "busy" in {n.id for n in ov.graph.nodes}
    assert "36 nodes" in ov.graph.cut or "nodes are not shown" in ov.graph.cut


# G-06
def test_same_rows_give_identical_output_in_any_order():
    rows = fixture()
    expected = g.as_dict(g.build(**rows, include_history=True))
    rng = random.Random(14)
    for _ in range(5):
        shuffled = {k: (list(v) if isinstance(v, list) else v) for k, v in rows.items()}
        for k, v in shuffled.items():
            if isinstance(v, list):
                rng.shuffle(v)
        assert g.as_dict(g.build(**shuffled, include_history=True)) == expected
    assert g.as_dict(g.build(**rows, include_history=True)) == expected


def test_built_lists_are_sorted():
    graph = g.build(**fixture(), include_history=True)
    assert list(graph.nodes) == sorted(graph.nodes, key=g._node_key)
    assert list(graph.edges) == sorted(graph.edges, key=g._edge_key)


def test_schema_version_travels_with_every_graph():
    graph = g.build()
    assert graph.graph_schema_version == g.GRAPH_SCHEMA_VERSION == 1
    assert g.as_dict(graph)["graph_schema_version"] == 1
    assert g.local(g.build(**fixture()), "obs:2").graph_schema_version == 1
    assert g.overview(graph).graph.graph_schema_version == 1


def test_empty_input_is_a_valid_empty_graph():
    graph = g.build()
    assert graph.nodes == () and graph.edges == () and not graph.truncated
    assert g.overview(graph).groups == ()


# G-07 (opaque ids)
def test_opaque_ids_are_stable_prefixed_and_distinct_across_keys():
    a = g.opaque_id(KEY, NodeType.OBSERVATION, 42)
    assert a == g.opaque_id(KEY, NodeType.OBSERVATION, 42)
    assert a.startswith("obs_") and len(a) == len("obs_") + 16
    int(a[4:], 16)
    assert a != g.opaque_id(b"j" * 32, NodeType.OBSERVATION, 42)
    assert a != g.opaque_id(KEY, NodeType.OBSERVATION, 43)
    assert a != g.opaque_id(KEY, NodeType.PANEL, 42)  # same row id, different type
    prefixes = {
        NodeType.PATIENT: "pt_", NodeType.OBSERVATION: "obs_", NodeType.PANEL: "pnl_",
        NodeType.EXTRACTION: "ext_", NodeType.SOURCE_DOCUMENT: "doc_", NodeType.MEDICATION_PLAN: "med_",
        NodeType.MEDICATION_COURSE: "crs_", NodeType.DEVICE_STREAM: "dev_",
    }  # fmt: skip
    for t, p in prefixes.items():
        assert g.opaque_id(KEY, t, "x").startswith(p)


def test_opaque_concept_keeps_its_series_id_and_an_empty_key_is_refused():
    assert g.opaque_id(KEY, NodeType.CONCEPT, "loinc:2093-3") == "loinc:2093-3"
    with pytest.raises(ValueError):
        g.opaque_id(b"", NodeType.OBSERVATION, 1)


def test_opaque_graph_hides_row_ids_and_maps_back():
    graph = g.build(**fixture())
    hidden, back = g.opaque_graph(graph, KEY)
    assert len(hidden.nodes) == len(graph.nodes) and len(hidden.edges) == len(graph.edges)
    text = repr(g.as_dict(hidden))
    assert "obs:" not in text and "files/a1" not in text and "pnl:" not in text
    assert {back[n.id] for n in hidden.nodes} == {n.id for n in graph.nodes}
    again, _ = g.opaque_graph(graph, KEY)
    assert again == hidden
    other, _ = g.opaque_graph(graph, b"z" * 32)
    assert {n.id for n in other.nodes if n.type is not NodeType.CONCEPT}.isdisjoint(
        {n.id for n in hidden.nodes if n.type is not NodeType.CONCEPT}
    )

