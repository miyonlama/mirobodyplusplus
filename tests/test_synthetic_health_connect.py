"""The synthetic Health Connect fixtures are deterministic, signed as LDC documents, schema-valid and self-consistent."""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("synthetic_health_connect", ROOT / "scripts" / "synthetic_health_connect.py")
shc = importlib.util.module_from_spec(_spec)
sys.modules["synthetic_health_connect"] = shc
_spec.loader.exec_module(shc)

SCHEMA = ROOT / "tests" / "fixtures" / "ldc" / "webhook-schema.json"


@pytest.fixture(scope="module")
def fixtures() -> dict[str, bytes]:
    return shc.generate(20251102, shc.DEFAULT_TZ, date.fromisoformat(shc.DEFAULT_START), 30, volume=False)


def _payloads(files: dict[str, bytes]) -> dict[str, dict]:
    return {k: json.loads(v) for k, v in files.items() if k.endswith(".json") and not k.endswith(("expectations.json", "index.json"))}


def test_same_seed_is_byte_identical(fixtures):
    again = shc.generate(20251102, shc.DEFAULT_TZ, date.fromisoformat(shc.DEFAULT_START), 30)
    assert again == fixtures


def test_other_seed_differs(fixtures):
    other = shc.generate(1, shc.DEFAULT_TZ, date.fromisoformat(shc.DEFAULT_START), 30)
    assert other["baseline_30_days/01.json"] != fixtures["baseline_30_days/01.json"]


def test_written_files_match_generated_bytes(tmp_path, fixtures):
    shc.write(fixtures, tmp_path)
    for rel, data in fixtures.items():
        assert (tmp_path / rel).read_bytes() == data


def test_signature_vector():
    body = b'{"test":true}\n'
    expected = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
    assert shc.sign_body("  test-secret\n", body) == expected  # surrounding whitespace is not part of the key
    # RFC 4231 test case 2 with a fixed message pins the algorithm itself.
    assert shc.sign_body("Jefe", b"what do ya want for nothing?") == (
        "sha256=5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
    )


def test_every_payload_validates_against_ldc_schema(fixtures):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft7Validator(schema)
    payloads = _payloads(fixtures)
    assert payloads
    for name, payload in payloads.items():
        errors = [e.message[:120] for e in validator.iter_errors(payload)]
        assert not errors, (name, errors[:3])


def test_expectations_reference_only_uuids_in_payloads(fixtures):
    index = json.loads(fixtures["index.json"])
    for scenario in index["scenarios"]:
        exp = json.loads(fixtures[f"{scenario}/expectations.json"])
        present = set()
        for fname in exp["delivery_order"]:
            payload = json.loads(fixtures[f"{scenario}/{fname}"])
            for key, value in payload.items():
                if isinstance(value, list) and key not in ("deleted_records", "daily_totals"):
                    present.update(i["uuid"] for i in value if isinstance(i, dict) and "uuid" in i)
        assert set(exp["retracted_uuids"]) <= present, scenario
        assert set(exp["final_values"]) <= present, scenario


def test_scenario_outcomes_are_the_documented_ones(fixtures):
    def exp(name):
        return json.loads(fixtures[f"{name}/expectations.json"])

    changed = exp("changed_value")
    assert [v["value"] for v in changed["final_values"].values()] == [1250]
    assert changed["accepted"] == {"steps": 1}

    gone = exp("deleted_record")
    assert len(gone["retracted_uuids"]) == 1 and gone["accepted"]["weight"] == 2 and gone["live"]["weight"] == 1

    samples = exp("per_sample_deletion")
    assert len(samples["retracted_uuids"]) == 5 and samples["live"]["heart_rate"] == 3
    assert all("#" in u for u in samples["retracted_uuids"])

    totals = exp("daily_totals_out_of_order")
    (winner,) = totals["daily_totals_winner"].values()
    assert winner["steps"] == 9000
    first = json.loads(fixtures["daily_totals_out_of_order/" + totals["delivery_order"][0]])
    assert first["sequence"] == winner["sequence"], "the winning payload is delivered first"

    assert exp("reproductive_health")["refused"] == {shc.REASON_NOT_ENABLED: {"menstruation_flow": 1}}
    assert exp("reproductive_health")["accepted"] == {}
    unknown = exp("unknown_key")
    assert unknown["refused"] == {shc.REASON_UNMAPPED: {"future_vital": 1}} and unknown["accepted"] == {"steps": 1}
    assert exp("bucketed_steps")["refused"] == {shc.REASON_BUCKETED: {"steps": 3}}
    assert exp("test_ping")["ping"] is True


def test_baseline_covers_thirty_days_and_the_dst_day(fixtures):
    base = json.loads(fixtures["baseline_30_days/expectations.json"])
    assert base["accepted"]["steps"] == 30 * 4
    assert base["accepted"]["heart_rate"] == 30 * 120
    assert len(base["daily_totals_winner"]) == 30
    (change,) = base["dst"]["transitions"]
    assert change["date"] == "2025-03-30" and change["local_day_hours"] == 23
    assert change["sleep_seconds_across_transition"] == 7 * 3600  # 23:00 to 07:00 wall clock is 7 h that night


def test_backfill_marks_only_the_last_payload_complete(fixtures):
    flags = [json.loads(fixtures[f"backfill_72h/{f}"]) for f in json.loads(fixtures["backfill_72h/expectations.json"])["delivery_order"]]
    assert [p["window_complete"] for p in flags] == [False, False, True]
    assert all(p["backfill"] is True for p in flags)


def test_heart_rate_sample_uuids_are_record_id_and_epoch_ms(fixtures):
    payload = json.loads(fixtures["baseline_30_days/01.json"])
    for sample in payload["heart_rate"][:5]:
        record_id, ms = sample["uuid"].split("#")
        assert len(record_id) == 36 and int(ms) % 1000 == 0


def test_volume_scenario_respects_ldc_record_cap():
    files = shc.generate(3, shc.DEFAULT_TZ, date.fromisoformat(shc.DEFAULT_START), 1, volume=True)
    exp = json.loads(files["heart_rate_1hz_day/expectations.json"])
    assert exp["accepted"]["heart_rate"] == 86400
    assert all(len(json.loads(files[f"heart_rate_1hz_day/{f}"])["heart_rate"]) <= 1000 for f in exp["delivery_order"])


def test_builtin_eu_zone_matches_the_dst_rule():
    zone = shc._EuZone(1)
    from datetime import datetime

    assert datetime(2025, 3, 29, 12, tzinfo=zone).utcoffset().total_seconds() == 3600
    assert datetime(2025, 3, 30, 12, tzinfo=zone).utcoffset().total_seconds() == 7200
    assert datetime(2025, 10, 26, 12, tzinfo=zone).utcoffset().total_seconds() == 3600
