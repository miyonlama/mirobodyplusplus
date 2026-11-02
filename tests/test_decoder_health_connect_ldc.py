"""Life Dashboard Companion payloads decode to catalogue facts, or are refused with a reason.

Payloads are invented; the shapes are those of
https://github.com/owen282000/life-dashboard-companion-app/blob/main/docs/webhook.md
(read 2026-11-02).
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from importlib import resources

import pytest

from mirobody.kernel import decoders, metrics
from mirobody.kernel.decoders import health_connect_ldc as ldc

#: Every record key webhook.md documents, by section (34). A new key upstream
#: is a new line here and a new row in the TSV.
DOCUMENTED_KEYS = frozenset(
    "steps distance active_calories total_calories speed exercise "
    "weight height body_temperature "
    "body_fat lean_body_mass bone_mass body_water_mass "
    "heart_rate resting_heart_rate heart_rate_variability blood_pressure blood_glucose oxygen_saturation respiratory_rate "
    "sleep hydration nutrition mindfulness "
    "menstruation_period menstruation_flow intermenstrual_bleeding ovulation_test cervical_mucus sexual_activity "
    "basal_body_temperature basal_metabolic_rate vo2_max skin_temperature".split()
)
REPRODUCTIVE = frozenset(
    "menstruation_period menstruation_flow intermenstrual_bleeding ovulation_test cervical_mucus sexual_activity "
    "basal_body_temperature".split()
)
U1 = "eb52b2c4-e3f7-49f3-8f92-34ca86640305"


def payload(**records) -> dict:
    return {"timestamp": "2026-10-04T07:15:02.481Z", "app_version": "1.23.0", "source": "health_connect",
            "sequence": 7, "timezone": "Europe/Amsterdam", **records}


def ms(text: str) -> int:
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)


def one(p: dict, metric: str):
    rows = [r for r in ldc.decode_rows(p, "UTC") if r.fact.metric_key == metric]
    assert len(rows) == 1, rows
    return rows[0]


# -- coverage ----------------------------------------------------------------


def test_every_documented_key_maps_or_is_declined_with_a_reason():
    assert set(ldc.CROSSWALK) - {"daily_totals"} == DOCUMENTED_KEYS
    for key, rows in ldc.CROSSWALK.items():
        assert any(not c.declined for c in rows) or all(c.reason == "no-metric" and c.note for c in rows), key
        for c in rows:
            assert c.declined == bool(c.reason), (key, c.ldc_field)
            if c.declined:
                assert c.note, (key, c.ldc_field)
            else:
                assert c.metric in metrics.METRICS, c.metric
                assert c.src_unit
                target = metrics.METRICS[c.metric].unit_ucum
                assert ldc._to_catalogue_unit(1.0, c.src_unit, c.metric) is not None, (c.metric, c.src_unit, target)


def test_reproductive_keys_are_exactly_the_gated_ones():
    gated = {k for k, rows in ldc.CROSSWALK.items() if rows[0].gate == "reproductive"}
    assert gated == REPRODUCTIVE


def test_classes_agree_with_health_connect_crosswalk():
    text = resources.files("mirobody").joinpath("res", "crosswalks", "health_connect.tsv").read_text(encoding="utf-8")
    body = "".join(line for line in io.StringIO(text) if not line.startswith("#"))
    hc: dict[str, set[str]] = {}
    for r in csv.DictReader(io.StringIO(body), delimiter="\t", quoting=csv.QUOTE_NONE):
        if r["metric"]:
            hc.setdefault(r["type"], set()).add(r["metric"])
    # Deliberate departures, each argued in the TSV note: the daily* sleep
    # aggregates are replaced by interval metrics, total calories is declined.
    departures = {"SleepSessionRecord", "TotalCaloriesBurnedRecord", "BasalMetabolicRateRecord", "SkinTemperatureRecord"}
    for rows in ldc.CROSSWALK.values():
        for c in rows:
            if c.hc_class in hc and c.metric and c.hc_class not in departures and c.ldc_key != "daily_totals" and c.ldc_field != "diastolic":
                assert c.metric in hc[c.hc_class], (c.ldc_key, c.metric, hc[c.hc_class])


def test_coverage_is_published_like_other_decoders():
    assert "health_connect_ldc" in decoders.DECODERS
    cov = decoders.coverage_of("health_connect_ldc")
    assert {"heartRates", "systolicPressures", "diastolicPressures", "vo2Maxs"} <= cov.timeseries
    assert cov.data_types == frozenset({"payload"})
    assert "distance.meters" in ldc.declined_keys()


def test_unmapped_key_is_refused_and_counted_not_guessed():
    p = payload(blood_alcohol=[{"x": 1, "time": "2026-10-04T07:00:00Z", "uuid": "u"}] * 2)
    assert ldc.decode_rows(p, "UTC") == []
    assert ldc.refusals(p) == [("blood_alcohol", "unmapped-key", 2)]


def test_declined_key_is_refused_no_metric():
    p = payload(distance=[{"meters": 1523.5, "start_time": "2026-10-04T08:00:00Z", "end_time": "2026-10-04T09:00:00Z", "uuid": U1}],
                basal_metabolic_rate=[{"kilocalories_per_day": 1650.0, "time": "2026-10-04T00:00:00Z", "uuid": "b"}])
    assert ldc.decode_rows(p, "UTC") == []
    assert ldc.refusals(p) == [("basal_metabolic_rate", "no-metric", 1), ("distance", "no-metric", 1)]


def test_skin_temperature_baseline_is_not_a_reading():
    p = payload(skin_temperature=[{"delta_celsius": -0.3, "baseline_celsius": 33.5, "time": "2026-10-04T03:10:00Z",
                                   "uuid": "01f1561e-0590-4560-9e3d-152b3a4d6048#1791083400000"}])
    rows = ldc.decode_rows(p, "UTC")
    assert [r.fact.metric_key for r in rows] == ["temperatureDelta"]
    assert rows[0].fact.value_num == pytest.approx(-0.3)
    assert ldc.refusals(p) == [("skin_temperature.baseline_celsius", "no-metric", 1)]


# -- unit conversion vectors -------------------------------------------------


@pytest.mark.parametrize(("key", "field", "value", "extra", "metric", "expected"), [
    ("weight", "kilograms", 75.5, {}, "bodyMasss", 75.5),
    ("height", "meters", 1.82, {}, "heights", 1.82),
    ("blood_glucose", "mmol_per_liter", 5.5, {}, "bloodGlucoseMoles", 5.5),
    ("oxygen_saturation", "percentage", 98.0, {}, "oxygenSaturations", 98.0),
    ("body_fat", "percentage", 18.5, {}, "bodyFatPercentages", 18.5),
    ("body_temperature", "celsius", 36.6, {}, "bodyTemperatures", 36.6),
    ("respiratory_rate", "rate", 16.0, {}, "respiratoryRates", 16.0),
    ("heart_rate_variability", "heart_rate_variability_millis", 42.5, {}, "hrvRMSSD", 42.5),
    ("lean_body_mass", "kilograms", 61.5, {}, "bodyFatFreeWeight", 61.5),
    ("bone_mass", "kilograms", 3.2, {}, "bodyBone", 3.2),
    ("body_water_mass", "kilograms", 42.0, {}, "bodyWaterMass", 42.0),
    ("resting_heart_rate", "bpm", 58, {}, "restingHeartRates", 58.0),
])
def test_instant_values_land_in_the_catalogue_unit(key, field, value, extra, metric, expected):
    p = payload(**{key: [{field: value, "time": "2026-10-04T07:00:00Z", "uuid": "r1", **extra}]})
    row = one(p, metric)
    assert row.fact.value_num == pytest.approx(expected)
    assert row.fact.unit == metrics.METRICS[metric].unit_ucum
    assert row.fact.effective_start_ms == ms("2026-10-04T07:00:00Z")
    assert row.fact.effective_end_ms == 0
    assert row.fact.source_record_id == "r1"


def test_vo2_max_carries_through_the_unit_engine_not_a_factor():
    # The catalogue declares mL/min/kg; LDC says mL/kg/min. Same quantity, two
    # spellings of the unit: a value that is 1000x out would be the trap.
    row = one(payload(vo2_max=[{"vo2_ml_per_min_per_kg": 45.0, "time": "2026-10-04T09:00:00Z", "uuid": "v"}]), "vo2Maxs")
    assert metrics.METRICS["vo2Maxs"].unit_ucum == "mL/min/kg"
    assert row.fact.value_num == 45.0 and row.fact.unit == "mL/min/kg"
    assert row.vendor_field == "Vo2MaxRecord"


def test_durations_and_intervals_convert_seconds():
    p = payload(
        exercise=[{"type": "56", "name": "running", "start_time": "2026-10-04T07:00:00Z",
                   "end_time": "2026-10-04T08:00:00Z", "duration_seconds": 3600, "uuid": "e"}],
        mindfulness=[{"start_time": "2026-10-04T06:00:00Z", "end_time": "2026-10-04T06:15:00Z", "duration_seconds": 900, "uuid": "m"}],
        steps=[{"count": 412, "start_time": "2026-10-04T06:00:00Z", "end_time": "2026-10-04T07:00:00Z", "uuid": U1,
                "source": "com.fitbit.FitbitMobile"}],
        hydration=[{"liters": 0.5, "start_time": "2026-10-04T08:00:00Z", "end_time": "2026-10-04T08:00:00Z", "uuid": "h"}],
    )
    assert one(p, "workoutDuration").fact.value_num == pytest.approx(60.0)
    assert one(p, "mindfulnessMinutes").fact.value_num == pytest.approx(15.0)
    steps = one(p, "steps")
    assert steps.fact.value_num == 412 and steps.fact.is_interval
    assert steps.fact.series_key == "hc:com.fitbit.FitbitMobile"
    assert steps.vendor_field == "StepsRecord"
    water = one(p, "waters")
    assert water.fact.value_num == 0.5 and not water.fact.is_interval  # an empty-length interval is an instant


def test_fractional_seconds_of_every_width_parse():
    for t in ("2026-10-04T08:41:17Z", "2026-10-04T08:41:17.203Z", "2026-10-04T08:41:17.203456Z", "2026-10-04T08:41:17.203456789Z"):
        assert ldc._ms(t) >= ms("2026-10-04T08:41:17Z")
    assert ldc._ms("2026-10-04T08:41:17.203456789Z") == ms("2026-10-04T08:41:17Z") + 203
    assert ldc._ms("2026-10-04T08:41:17") == 0 and ldc._ms("nonsense") == 0


# -- blood pressure ----------------------------------------------------------


def test_blood_pressure_splits_into_two_rows_sharing_time_uuid_and_panel():
    p = payload(blood_pressure=[{"systolic": 120.0, "diastolic": 80.0, "time": "2026-10-04T07:00:00Z", "uuid": "bp1"}])
    rows = ldc.decode_rows(p, "UTC")
    by = {r.fact.metric_key: r for r in rows}
    assert set(by) == {"systolicPressures", "diastolicPressures"}
    assert by["systolicPressures"].fact.value_num == 120.0 and by["diastolicPressures"].fact.value_num == 80.0
    assert {r.fact.effective_start_ms for r in rows} == {ms("2026-10-04T07:00:00Z")}
    assert {r.record_uuid for r in rows} == {"bp1"}
    assert len({r.fact.panel_id for r in rows}) == 1 and rows[0].fact.panel_id == "bp1"
    assert {r.vendor_field for r in rows} == {"BloodPressureRecord"}


def test_blood_pressure_missing_a_half_keeps_the_other_and_counts_the_gap():
    p = payload(blood_pressure=[{"systolic": 120.0, "time": "2026-10-04T07:00:00Z", "uuid": "bp1"}])
    assert [r.fact.metric_key for r in ldc.decode_rows(p, "UTC")] == ["systolicPressures"]
    assert ldc.refusals(p) == [("blood_pressure.diastolic", "malformed", 1)]


# -- per-sample uuids --------------------------------------------------------


def test_samples_keep_their_full_uuid_so_identities_do_not_collide():
    base = "708a76cb-45d3-45ea-9f8b-26d6d55d2692"
    p = payload(heart_rate=[
        {"bpm": 61, "time": "2026-10-04T06:58:00Z", "uuid": f"{base}#1791097080000", "source": "com.x"},
        {"bpm": 63, "time": "2026-10-04T06:59:00Z", "uuid": f"{base}#1791097140000", "source": "com.x"},
    ])
    rows = ldc.decode_rows(p, "UTC")
    assert [r.fact.source_record_id for r in rows] == [f"{base}#1791097080000", f"{base}#1791097140000"]
    assert {r.vendor_field for r in rows} == {"HeartRateRecord"}
    assert rows[0].fact.metric_key == "heartRates" and rows[0].fact.value_num == 61.0


def test_a_record_without_uuid_or_time_is_malformed_not_stamped_now():
    p = payload(heart_rate=[{"bpm": 61, "time": "2026-10-04T06:58:00Z"}, {"bpm": 62, "uuid": "u"}, {"bpm": "x", "time": "2026-10-04T06:58:00Z", "uuid": "w"}])
    assert ldc.decode_rows(p, "UTC") == []
    assert ldc.refusals(p) == [("heart_rate", "malformed", 2), ("heart_rate.bpm", "malformed", 1)]


def test_sleep_session_and_stages():
    p = payload(sleep=[{
        "session_end_time": "2026-10-04T06:30:00Z", "duration_seconds": 28800, "uuid": "s1",
        "stages": [
            {"stage": "deep", "start_time": "2026-10-03T23:00:00Z", "end_time": "2026-10-04T01:00:00Z", "duration_seconds": 7200},
            {"stage": "light", "start_time": "2026-10-04T01:00:00Z", "end_time": "2026-10-04T02:00:00Z", "duration_seconds": 3600},
            {"stage": "light", "start_time": "2026-10-04T03:00:00Z", "end_time": "2026-10-04T04:00:00Z", "duration_seconds": 3600},
            {"stage": "out_of_bed", "start_time": "2026-10-04T04:00:00Z", "end_time": "2026-10-04T04:10:00Z", "duration_seconds": 600},
        ]}])
    rows = ldc.decode_rows(p, "UTC")
    session = next(r for r in rows if r.fact.metric_key == "sleepAnalysis_InBed")
    assert session.fact.effective_start_ms == ms("2026-10-03T22:30:00Z") and session.fact.effective_end_ms == ms("2026-10-04T06:30:00Z")
    assert session.fact.value_num == 28800 * 1000 and session.fact.unit == "ms"
    deep = next(r for r in rows if r.fact.metric_key == "sleepAnalysis_Asleep(Deep)")
    assert deep.fact.value_num == 7200 * 1000
    lights = [r for r in rows if r.fact.metric_key == "sleepAnalysis_Asleep(Core)"]
    assert len({r.fact.source_record_id for r in lights}) == 2 and all(r.record_uuid == "s1" for r in lights)
    assert ldc.refusals(p) == [("sleep.stages:out_of_bed", "no-metric", 1)]


# -- gates and refusals ------------------------------------------------------


def reproductive_payload() -> dict:
    return payload(
        basal_body_temperature=[{"celsius": 36.4, "time": "2026-10-12T06:30:00Z", "uuid": "bbt"}],
        menstruation_flow=[{"flow": "medium", "time": "2026-10-03T00:00:00Z", "uuid": "mf"}],
        sexual_activity=[{"protection_used": "protected", "time": "2026-10-11T00:00:00Z", "uuid": "sa"}],
    )


def test_reproductive_keys_are_refused_unless_opted_in():
    p = reproductive_payload()
    assert ldc.decode_rows(p, "UTC") == []
    assert ldc.refusals(p) == [("basal_body_temperature", "type-not-enabled", 1),
                               ("menstruation_flow", "type-not-enabled", 1), ("sexual_activity", "type-not-enabled", 1)]
    assert ldc.decode("payload", p, "UTC") == []


def test_opting_in_decodes_only_what_the_catalogue_can_hold():
    p = reproductive_payload()
    rows = ldc.decode_rows(p, "UTC", accept_reproductive=True)
    assert [r.fact.metric_key for r in rows] == ["reproductiveBasalBodyTemperature"]
    assert rows[0].fact.value_num == pytest.approx(36.4)
    assert ldc.refusals(p, accept_reproductive=True) == [("menstruation_flow", "no-metric", 1), ("sexual_activity", "no-metric", 1)]


def test_healthkit_ios_payload_is_refused_whole_because_hrv_is_sdnn_there():
    p = payload(heart_rate_variability=[{"heart_rate_variability_millis": 48.0, "time": "2026-10-04T03:10:00Z", "uuid": "h"}],
                steps=[{"count": 10, "start_time": "2026-10-04T06:00:00Z", "end_time": "2026-10-04T07:00:00Z", "uuid": "s"}])
    p["source"] = "healthkit_ios"
    assert ldc.decode_rows(p, "UTC") == [] and ldc.decode("payload", p, "UTC") == []
    assert ldc.refusals(p) == [("heart_rate_variability", "wrong-source", 1), ("steps", "wrong-source", 1)]
    p["source"] = "screen_time"
    assert ldc.decode_rows(p, "UTC") == []


def test_bucketed_series_is_refused_other_keys_still_decode():
    p = payload(
        heart_rate=[{"bucket_start": "2026-10-04T08:00:00Z", "bucket_end": "2026-10-04T08:01:00Z", "sample_count": 58,
                     "avg": 72.4, "min": 66.0, "max": 81.0}],
        resting_heart_rate=[{"bpm": 58, "time": "2026-10-04T07:00:00Z", "uuid": "r"}],
    )
    p["_resolutions"] = {"heart_rate": "1m"}
    assert [r.fact.metric_key for r in ldc.decode_rows(p, "UTC")] == ["restingHeartRates"]
    assert ldc.refusals(p) == [("heart_rate", "bucketed-unsupported", 1)]
    del p["_resolutions"]  # a bucket without the declaration is still not a reading
    assert ldc.refusals(p) == [("heart_rate", "bucketed-unsupported", 1)]


def test_test_ping_and_non_dict_decode_to_nothing():
    assert ldc.decode_rows({"test": True, "source": "health_connect", "message": "x"}, "UTC") == []
    assert ldc.refusals({"test": True, "source": "health_connect"}) == []
    assert ldc.decode_rows([], "UTC") == [] and ldc.decode("sleep", {}, "UTC") == []


# -- deletions ---------------------------------------------------------------


def test_deletions_name_the_key_the_uuid_and_whether_samples_follow():
    p = {"deleted_records": [
        {"type": "nutrition", "uuid": "n1"},
        {"type": "weight", "uuid": "w1"},
        {"type": "heart_rate", "uuid": "h1"},
        {"type": "sleep", "uuid": "s1"},
        {"type": "skin_temperature", "uuid": "t1"},
        {"type": "weight", "uuid": "w1"},
        {"type": "who_knows", "uuid": "x"},
        {"type": "speed"},
    ]}
    assert ldc.deletions(p) == [("weight", "w1", False), ("heart_rate", "h1", True), ("sleep", "s1", True),
                                ("skin_temperature", "t1", True)]
    assert ldc.deletions({}) == []


# -- time --------------------------------------------------------------------


def test_daily_totals_are_cut_in_the_payload_zone_and_survive_dst():
    p = payload(daily_totals=[
        {"date": "2026-10-24", "steps": 9874, "distance_meters": 7120.0, "active_calories": 498.4, "total_calories": 2390.1},
        {"date": "2026-10-25", "steps": 8421, "total_calories": 2231.5},  # Europe/Amsterdam falls back: 25 h
        {"date": "2026-03-29", "steps": 100},  # and springs forward: 23 h
    ])
    rows = ldc.decode_rows(p, "Asia/Shanghai")
    spans = {(r.fact.metric_key, r.fact.source_record_id): r.fact for r in rows}
    hour = 3600 * 1000
    assert (spans["steps", "daily_totals:2026-10-24:steps"].effective_end_ms - spans["steps", "daily_totals:2026-10-24:steps"].effective_start_ms) == 24 * hour
    f25 = spans["steps", "daily_totals:2026-10-25:steps"]
    assert f25.effective_end_ms - f25.effective_start_ms == 25 * hour
    assert f25.effective_start_ms == ms("2026-10-24T22:00:00Z")  # midnight CEST, not Shanghai's
    f29 = spans["steps", "daily_totals:2026-03-29:steps"]
    assert f29.effective_end_ms - f29.effective_start_ms == 23 * hour
    assert f25.statistic == "provider_daily"
    assert spans["dailyTotalCalories", "daily_totals:2026-10-25:total_calories"].value_num == 2231.5
    assert ("daily_totals.distance_meters", "no-metric", 1) in ldc.refusals(p)


def test_caller_zone_is_the_fallback_when_the_payload_has_none_or_a_bad_one():
    entry = [{"date": "2026-10-25", "steps": 1}]
    for tzfield in ({}, {"timezone": "Not/AZone"}):
        p = payload(daily_totals=entry)
        p.pop("timezone")
        p.update(tzfield)
        (r,) = ldc.decode_rows(p, "Europe/Amsterdam")
        assert r.fact.effective_end_ms - r.fact.effective_start_ms == 25 * 3600 * 1000


def test_instants_are_exact_utc_across_a_dst_change():
    # 00:30 and 02:30 UTC on the night Amsterdam changes clocks: wall-clock arithmetic would misplace one.
    p = payload(weight=[{"kilograms": 70.0, "time": "2026-10-25T00:30:00Z", "uuid": "a"},
                        {"kilograms": 70.5, "time": "2026-10-25T02:30:00Z", "uuid": "b"}])
    a, b = ldc.decode_rows(p, "Europe/Amsterdam")
    assert b.fact.effective_start_ms - a.fact.effective_start_ms == 2 * 3600 * 1000
    assert datetime.fromtimestamp(a.fact.effective_start_ms / 1000, UTC).hour == 0
