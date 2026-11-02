#!/usr/bin/env python3
"""Deterministic, synthetic Health Connect payloads in the Life Dashboard
Companion (LDC) webhook shape, plus the outcome a receiver must reach.

Every value is invented. Nothing here reads a database, a network or a clock:
the same ``--seed``, ``--tz``, ``--start-date`` and ``--days`` give
byte-identical files, so a diff after an upstream merge means the receiver
changed, not the fixtures.

    python scripts/synthetic_health_connect.py --out build/hc-fixtures
    python scripts/synthetic_health_connect.py --out build/hc-fixtures --seed 7 --volume
    python scripts/synthetic_health_connect.py --sign build/hc-fixtures/baseline_30_days/01.json --secret s3cret

Standard library only. ``--tz`` is resolved through ``zoneinfo``; on a machine
without tz data (stock Windows) the EU zones listed in ``_EU_STD_HOURS`` fall
back to a built-in rule that yields the same instants.

The default zone and start date put the Europe/Berlin spring-forward
(2025-03-30, 23 local hours) inside the 30 generated days, so a receiver that
cuts days on a fixed UTC offset gets a daily total wrong.

Layout of ``--out``::

    index.json                  seed, tz, app_version, scenario names in replay order
    <scenario>/NN.json          one LDC payload per file
    <scenario>/expectations.json

Payload files are what the phone would POST; sign their raw bytes with
``sign_body`` (``X-Signature: sha256=<hex>``). ``sequence`` is one counter for
the whole set, as it is for one install, so scenarios can be replayed together.

Expectations format (``format`` 1). Counts are of distinct ``uuid`` per LDC
key unless said otherwise; the file is computed by the reference model in
``expected_outcome`` from the payloads, never typed by hand.

``scenario``, ``description``, ``seed``, ``timezone``, ``app_version``
    Provenance. Fixtures are versioned by the LDC ``app_version`` they emulate.
``enabled_types``
    LDC keys the receiver is assumed to have switched on. A key outside this
    list and outside ``unmapped`` is refused as ``type-not-enabled``
    (reproductive health is opt-in and absent from the list).
``delivery_order``
    Payload files in the order the phone delivered them. ``sequence`` order and
    delivery order differ in ``daily_totals_out_of_order``. A replay in any
    other order must reach the same state.
``accepted``
    ``{key: n}`` distinct uuids a receiver must store, including ones later
    retracted.
``retracted_uuids``
    uuids that must end up retracted: named by ``deleted_records`` (a sample
    uuid ``<record id>#<epoch ms>`` is matched by the record id as prefix) in a
    payload whose ``sequence`` is higher than the one that delivered the record.
``live``
    ``accepted`` minus retractions, per key.
``refused``
    ``{reason: {key: n}}``, reason one of ``type-not-enabled``,
    ``unmapped-key``, ``bucketed-unsupported``; ``n`` counts array items
    (buckets carry no uuid).
``final_values``
    ``{uuid: {"key", "field", "value", "sequence"}}`` for every live record of
    a scalar key: the version from the highest ``sequence`` wins. Heart-rate
    samples are covered by counts and ``heart_rate_range`` only.
``daily_totals_winner``
    ``{date: {"sequence", "steps", ...}}``, the payload with the highest
    ``sequence`` that carried that date.
``ping``
    True when the only payload is a test ping: nothing stored, 2xx answered.
``dst``
    Zone offset changes at local midnights inside the range and the local
    hours of each such day, plus ``sleep_seconds_across_transition`` where the
    night spans it.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import random
import sys
import uuid as uuidlib
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path

APP_VERSION = "1.27.0"  # first LDC release that sends `timezone`
DEFAULT_TZ = "Europe/Berlin"
DEFAULT_START = "2025-03-07"
SOURCE_APP = "com.example.synthetic.tracker"

REASON_NOT_ENABLED = "type-not-enabled"
REASON_UNMAPPED = "unmapped-key"
REASON_BUCKETED = "bucketed-unsupported"

# Record keys the receiver maps, with the field holding the scalar to compare.
SCALAR_FIELD = {
    "steps": "count",
    "resting_heart_rate": "bpm",
    "heart_rate_variability": "heart_rate_variability_millis",
    "weight": "kilograms",
    "blood_glucose": "mmol_per_liter",
    "oxygen_saturation": "percentage",
    "sleep": "duration_seconds",
}
# Keys whose items hold more than one scalar; the first one is compared.
MULTI_FIELD = {"blood_pressure": "systolic"}
MAPPED_KEYS = frozenset(SCALAR_FIELD) | frozenset(MULTI_FIELD) | {"heart_rate", "menstruation_flow"}
ENABLED_TYPES = sorted(MAPPED_KEYS - {"menstruation_flow"})

# Top-level LDC fields that are not record arrays.
ENVELOPE_KEYS = frozenset(
    {
        "timestamp", "timezone", "app_version", "source", "device", "sequence", "deleted_records",
        "deletions_unavailable", "records_outside_window", "test", "message", "backfill", "window_start",
        "window_end", "window_complete", "_diagnostics", "writeback", "daily_totals", "_resolutions",
    }
)  # fmt: skip

_EU_STD_HOURS = {
    "Europe/Berlin": 1, "Europe/Paris": 1, "Europe/Amsterdam": 1, "Europe/Madrid": 1, "Europe/Rome": 1,
    "Europe/Vienna": 1, "Europe/Zurich": 1, "Europe/Athens": 2, "Europe/London": 0,
}  # fmt: skip


def sign_body(secret: str, body: bytes) -> str:
    """``X-Signature`` header value for ``body`` as LDC documents it.

    The key is the secret with surrounding whitespace stripped, as UTF-8; the
    message is the raw request bytes, signed before any JSON parsing.
    """
    digest = hmac.new(secret.strip().encode("utf-8"), body, hashlib.sha256).hexdigest()
    return "sha256=" + digest


def _last_sunday(year: int, month: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() + 1) % 7)


class _EuZone(tzinfo):
    """EU summer-time rule: last Sunday of March to last Sunday of October, 01:00 UTC."""

    def __init__(self, std_hours: int) -> None:
        self._std = timedelta(hours=std_hours)

    def _dst_bounds(self, year: int) -> tuple[datetime, datetime]:
        # Naive local bounds: spring forward at 01:00 UTC, fall back at 01:00 UTC
        # expressed in summer time (ambiguous hour resolves to summer, fold=0).
        start = datetime.combine(_last_sunday(year, 3), datetime.min.time()) + timedelta(hours=1) + self._std
        end = datetime.combine(_last_sunday(year, 10), datetime.min.time()) + timedelta(hours=1) + self._std
        return start, end + timedelta(hours=1)

    def dst(self, dt: datetime | None) -> timedelta:
        if dt is None:
            return timedelta(0)
        start, end = self._dst_bounds(dt.year)
        naive = dt.replace(tzinfo=None)
        return timedelta(hours=1) if start <= naive < end else timedelta(0)

    def utcoffset(self, dt: datetime | None) -> timedelta:
        return self._std + self.dst(dt)

    def tzname(self, dt: datetime | None) -> str:
        return "EU"


def load_zone(name: str) -> tzinfo:
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # missing tz data raises ZoneInfoNotFoundError, a KeyError subclass
        if name in _EU_STD_HOURS:
            return _EuZone(_EU_STD_HOURS[name])
        raise SystemExit(f"no tz data for {name!r}: pip install tzdata, or pick one of {sorted(_EU_STD_HOURS)}") from None


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


class Generator:
    def __init__(self, seed: int, tz_name: str, start: date, days: int) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.tz_name = tz_name
        self.tz = load_zone(tz_name)
        self.start = start
        self.days = days
        self.seq = 0

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def new_uuid(self) -> str:
        return str(uuidlib.UUID(int=self.rng.getrandbits(128), version=4))

    def local(self, d: date, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
        return datetime(d.year, d.month, d.day, hour, minute, second, tzinfo=self.tz)

    def day(self, i: int) -> date:
        return self.start + timedelta(days=i)

    def envelope(self, sent_at: datetime, **fields: object) -> dict:
        payload: dict = {
            "timestamp": iso(sent_at),
            "timezone": self.tz_name,
            "app_version": APP_VERSION,
            "source": "health_connect",
            "sequence": self.next_seq(),
        }
        payload.update(fields)
        return payload

    def hr_record(self, d: date, hour: int, minutes: int, step_s: int = 60) -> list[dict]:
        """Samples of one Health Connect HR record; each carries ``<record id>#<epoch ms>``."""
        record_id = self.new_uuid()
        begin = self.local(d, hour)
        base = self.rng.randint(58, 72) if hour < 9 else self.rng.randint(68, 95)
        out = []
        for k in range(0, minutes * 60, step_s):
            at = begin + timedelta(seconds=k)
            out.append(
                {
                    "bpm": max(40, base + self.rng.randint(-4, 6)),
                    "time": iso(at),
                    "source": SOURCE_APP,
                    "uuid": f"{record_id}#{epoch_ms(at)}",
                }
            )
        return out

    def day_records(self, d: date, i: int) -> dict[str, list[dict]]:
        rng = self.rng
        recs: dict[str, list[dict]] = {}
        steps = []
        for hour in (8, 12, 17, 20):
            steps.append(
                {
                    "count": rng.randint(300, 3200),
                    "start_time": iso(self.local(d, hour)),
                    "end_time": iso(self.local(d, hour, 59, 59)),
                    "source": SOURCE_APP,
                    "uuid": self.new_uuid(),
                }
            )
        recs["steps"] = steps
        # One-minute samples across the morning; the 1 Hz volume day is a separate scenario.
        recs["heart_rate"] = self.hr_record(d, 7, 120)
        recs["resting_heart_rate"] = [
            {"bpm": rng.randint(54, 64), "time": iso(self.local(d, 6)), "source": SOURCE_APP, "uuid": self.new_uuid()}
        ]
        recs["heart_rate_variability"] = [
            {
                "heart_rate_variability_millis": round(rng.uniform(28.0, 72.0), 1),
                "time": iso(self.local(d, 6, 30)),
                "source": SOURCE_APP,
                "uuid": self.new_uuid(),
            }
        ]
        # Night that ends on day d: bedtime 23:00 the evening before, wake 07:00 local.
        bed = self.local(d - timedelta(days=1), 23)
        wake = self.local(d, 7)
        # Stage bounds are cut from the UTC span, so on the spring-forward night
        # (7 h of wall clock between 23:00 and 07:00) they still tile the session.
        bed_utc = bed.astimezone(UTC)
        duration = int((wake.astimezone(UTC) - bed_utc).total_seconds())
        stages = []
        edge = 0
        names = ("light", "deep", "light", "rem", "awake", "light")
        cuts = (0.10, 0.30, 0.55, 0.70, 0.73, 1.0)
        for stage, cut in zip(names, cuts, strict=True):
            nxt = round(duration * cut / 60) * 60 if cut < 1.0 else duration
            stages.append(
                {
                    "stage": stage,
                    "start_time": iso(bed_utc + timedelta(seconds=edge)),
                    "end_time": iso(bed_utc + timedelta(seconds=nxt)),
                    "duration_seconds": nxt - edge,
                }
            )
            edge = nxt
        recs["sleep"] = [
            {
                "session_end_time": iso(wake),
                "duration_seconds": duration,
                "stages": stages,
                "source": SOURCE_APP,
                "uuid": self.new_uuid(),
            }
        ]
        recs["oxygen_saturation"] = [
            {
                "percentage": round(rng.uniform(95.0, 99.0), 1),
                "time": iso(self.local(d, 22)),
                "source": SOURCE_APP,
                "uuid": self.new_uuid(),
            }
        ]
        if i % 3 == 0:
            recs["weight"] = [
                {
                    "kilograms": round(74.0 + rng.uniform(-0.8, 0.8), 1),
                    "time": iso(self.local(d, 7, 15)),
                    "source": SOURCE_APP,
                    "uuid": self.new_uuid(),
                }
            ]
        if i % 2 == 0:
            recs["blood_pressure"] = [
                {
                    "systolic": rng.randint(112, 128),
                    "diastolic": rng.randint(70, 82),
                    "time": iso(self.local(d, 9)),
                    "source": SOURCE_APP,
                    "uuid": self.new_uuid(),
                }
            ]
        if i % 2 == 1:
            recs["blood_glucose"] = [
                {
                    "mmol_per_liter": round(rng.uniform(4.6, 6.2), 1),
                    "time": iso(self.local(d, 7, 5)),
                    "source": SOURCE_APP,
                    "uuid": self.new_uuid(),
                }
            ]
        return recs

    def window_dates(self, first: date, last: date) -> list[date]:
        return [first + timedelta(days=k) for k in range((last - first).days + 1)]

    def totals_entry(self, d: date, steps: int) -> dict:
        return {
            "date": d.isoformat(),
            "steps": steps,
            "distance_meters": round(steps * 0.74, 1),
            "active_calories": round(steps * 0.04, 1),
            "total_calories": round(1750 + steps * 0.04, 1),
        }

    # ---- scenarios: each returns (description, [payload, ...] in delivery order) ----

    def baseline(self) -> tuple[str, list[dict]]:
        payloads = []
        for i in range(self.days):
            d = self.day(i)
            recs = self.day_records(d, i)
            steps_total = sum(r["count"] for r in recs["steps"])
            sent = self.local(d, 23, 30)
            payloads.append(self.envelope(sent, daily_totals=[self.totals_entry(d, steps_total)], **recs))
        return "Thirty local days of every mapped vital, one payload per day, in order.", payloads

    def backfill_72h(self) -> tuple[str, list[dict]]:
        # The window ends at the start of day 0 and runs 72 h back; bounds are not at local midnight in general.
        end = self.local(self.day(0)).astimezone(UTC)
        begin = end - timedelta(hours=72)
        first_day = (begin.astimezone(self.tz)).date()
        payloads = []
        days = self.window_dates(first_day, self.day(0) - timedelta(days=1))
        for n, d in enumerate(days):
            recs = {k: v for k, v in self.day_records(d, n).items() if k in ("steps", "heart_rate", "weight")}
            last = n == len(days) - 1
            steps_total = sum(r["count"] for r in recs["steps"])
            payloads.append(
                self.envelope(
                    end + timedelta(minutes=5 + n),
                    backfill=True,
                    window_start=iso(begin),
                    window_end=iso(end),
                    window_complete=last,
                    daily_totals=[self.totals_entry(d, steps_total)],
                    **recs,
                )
            )
        return "One 72-hour backfill window; only the last payload carries window_complete=true.", payloads

    def changed_value(self) -> tuple[str, list[dict]]:
        d = self.day(self.days - 1) + timedelta(days=1)
        rid = self.new_uuid()
        record = {
            "count": 1000,
            "start_time": iso(self.local(d, 10)),
            "end_time": iso(self.local(d, 10, 59, 59)),
            "source": SOURCE_APP,
            "uuid": rid,
        }
        first = self.envelope(self.local(d, 11), steps=[record])
        revised = dict(record, count=1250)
        second = self.envelope(self.local(d, 12), steps=[revised])
        return "A step record resent with the same uuid and a higher sequence; 1250 wins.", [first, second]

    def deleted_record(self) -> tuple[str, list[dict]]:
        d = self.day(self.days - 1) + timedelta(days=1)
        rid = self.new_uuid()
        weight = {"kilograms": 73.4, "time": iso(self.local(d, 7)), "source": SOURCE_APP, "uuid": rid}
        keep = {"kilograms": 73.6, "time": iso(self.local(d, 8)), "source": SOURCE_APP, "uuid": self.new_uuid()}
        first = self.envelope(self.local(d, 9), weight=[weight, keep])
        second = self.envelope(self.local(d, 10), deleted_records=[{"type": "weight", "uuid": rid}])
        return "A weight record, then a deleted_records entry naming it; its sibling stays.", [first, second]

    def per_sample_deletion(self) -> tuple[str, list[dict]]:
        d = self.day(self.days - 1) + timedelta(days=1)
        samples = self.hr_record(d, 13, 5)
        record_id = samples[0]["uuid"].split("#")[0]
        other = self.hr_record(d, 15, 3)
        first = self.envelope(self.local(d, 16), heart_rate=samples + other)
        second = self.envelope(self.local(d, 17), deleted_records=[{"type": "heart_rate", "uuid": record_id}])
        return "Five heart-rate samples of one record deleted by the record id (prefix match).", [first, second]

    def daily_totals_out_of_order(self) -> tuple[str, list[dict]]:
        d = self.day(self.days - 1) + timedelta(days=2)
        older = self.envelope(self.local(d, 12), daily_totals=[self.totals_entry(d, 7000)])
        newer = self.envelope(self.local(d, 20), daily_totals=[self.totals_entry(d, 9000)])
        # The phone retried the older payload after the newer one: delivery order is reversed.
        return "Two daily_totals for one date, delivered newest sequence first; the highest sequence wins.", [
            newer,
            older,
        ]

    def reproductive(self) -> tuple[str, list[dict]]:
        d = self.day(5)
        record = {"flow": "medium", "time": iso(self.local(d, 8)), "source": SOURCE_APP, "uuid": self.new_uuid()}
        return "A menstruation_flow record; reproductive health is not an enabled type.", [
            self.envelope(self.local(d, 9), menstruation_flow=[record])
        ]

    def unknown_key(self) -> tuple[str, list[dict]]:
        d = self.day(6)
        steps = {
            "count": 640,
            "start_time": iso(self.local(d, 14)),
            "end_time": iso(self.local(d, 14, 59, 59)),
            "source": SOURCE_APP,
            "uuid": self.new_uuid(),
        }
        unknown = [{"level": 3, "time": iso(self.local(d, 14)), "source": SOURCE_APP, "uuid": self.new_uuid()}]
        return "A key LDC may add later next to a known one: the known one is stored, the key is named, not dropped.", [
            self.envelope(self.local(d, 15), steps=[steps], future_vital=unknown)
        ]

    def bucketed(self) -> tuple[str, list[dict]]:
        d = self.day(7)
        buckets = []
        for k in range(3):
            begin = self.local(d, 9 + k)
            buckets.append(
                {
                    "bucket_start": iso(begin),
                    "bucket_end": iso(begin + timedelta(hours=1)),
                    "sample_count": 4,
                    "total": float(self.rng.randint(200, 900)),
                    "complete": True,
                }
            )
        return "Steps sent as one-hour buckets (data resolution on): no uuid, refused for v1.", [
            self.envelope(self.local(d, 13), steps=buckets, _resolutions={"steps": "1h"})
        ]

    def test_ping(self) -> tuple[str, list[dict]]:
        sent = self.local(self.start)
        return "The Test ping button: no records, no sequence.", [
            {
                "timestamp": iso(sent),
                "app_version": APP_VERSION,
                "source": "health_connect",
                "test": True,
                "message": "Test ping from Life Dashboard Companion",
            }
        ]

    def heart_rate_1hz_day(self) -> tuple[str, list[dict]]:
        d = self.start
        samples = self.hr_record(d, 0, 24 * 60, step_s=1)
        payloads = []
        for n in range(0, len(samples), 1000):  # LDC caps heart_rate at 1000 records per payload
            payloads.append(self.envelope(self.local(d, 23, 59), heart_rate=samples[n : n + 1000]))
        return "One local day of 1 Hz heart-rate samples, split at LDC's 1000-record cap (volume).", payloads


SCENARIOS = [
    "baseline_30_days",
    "backfill_72h",
    "changed_value",
    "deleted_record",
    "per_sample_deletion",
    "daily_totals_out_of_order",
    "reproductive_health",
    "unknown_key",
    "bucketed_steps",
    "test_ping",
]
VOLUME_SCENARIO = "heart_rate_1hz_day"
_BUILDERS = {
    "baseline_30_days": "baseline",
    "backfill_72h": "backfill_72h",
    "changed_value": "changed_value",
    "deleted_record": "deleted_record",
    "per_sample_deletion": "per_sample_deletion",
    "daily_totals_out_of_order": "daily_totals_out_of_order",
    "reproductive_health": "reproductive",
    "unknown_key": "unknown_key",
    "bucketed_steps": "bucketed",
    "test_ping": "test_ping",
    VOLUME_SCENARIO: "heart_rate_1hz_day",
}


def _scalar(key: str, item: dict) -> tuple[str, object]:
    field = SCALAR_FIELD.get(key) or MULTI_FIELD[key]
    return field, item[field]


def expected_outcome(payloads: list[dict]) -> dict:
    """Reference model of what a conforming receiver stores, independent of delivery order.

    Conflicts are settled by ``sequence`` alone, as LDC specifies: a late
    payload still contributes its records, a record stays deleted when a
    deletion with a higher sequence named it, and a date's daily totals come
    from the highest sequence that carried it.
    """
    versions: dict[tuple[str, str], tuple[int, dict]] = {}
    deletions: list[tuple[str, str, int]] = []
    refused: dict[str, dict[str, int]] = {}
    totals: dict[str, tuple[int, dict]] = {}
    for p in payloads:
        seq = p.get("sequence", 0)
        resolutions = p.get("_resolutions", {})
        for d in p.get("deleted_records", []):
            deletions.append((d["type"], d["uuid"], seq))
        for t in p.get("daily_totals", []):
            if t["date"] not in totals or totals[t["date"]][0] < seq:
                totals[t["date"]] = (seq, t)
        for key, value in p.items():
            if key in ENVELOPE_KEYS or not isinstance(value, list):
                continue
            if key in resolutions:
                reason = REASON_BUCKETED
            elif key not in MAPPED_KEYS:
                reason = REASON_UNMAPPED
            elif key not in ENABLED_TYPES:
                reason = REASON_NOT_ENABLED
            else:
                for item in value:
                    slot = (key, item["uuid"])
                    if slot not in versions or versions[slot][0] < seq:
                        versions[slot] = (seq, item)
                continue
            refused.setdefault(reason, {})
            refused[reason][key] = refused[reason].get(key, 0) + len(value)

    retracted = sorted(
        uid
        for (key, uid), (seq, _) in versions.items()
        if any(dk == key and (uid == du or uid.startswith(du + "#")) and ds > seq for dk, du, ds in deletions)
    )
    gone = set(retracted)
    accepted: dict[str, int] = {}
    live: dict[str, int] = {}
    final_values: dict[str, dict] = {}
    for (key, uid), (seq, item) in sorted(versions.items()):
        accepted[key] = accepted.get(key, 0) + 1
        if uid in gone:
            continue
        live[key] = live.get(key, 0) + 1
        if key != "heart_rate":
            field, value = _scalar(key, item)
            final_values[uid] = {"key": key, "field": field, "value": value, "sequence": seq}
    out = {
        "accepted": dict(sorted(accepted.items())),
        "retracted_uuids": retracted,
        "live": dict(sorted(live.items())),
        "refused": {r: dict(sorted(v.items())) for r, v in sorted(refused.items())},
        "final_values": final_values,
        "daily_totals_winner": {
            day: dict(t, sequence=seq) for day, (seq, t) in sorted(totals.items())
        },
        "ping": bool(payloads) and all(p.get("test") is True for p in payloads),
    }
    bpm = [item["bpm"] for (key, _), (_, item) in versions.items() if key == "heart_rate"]
    if bpm:
        out["heart_rate_range"] = {"min": min(bpm), "max": max(bpm)}
    return out


def dst_facts(gen: Generator) -> dict:
    """Local days whose length is not 24 h: the day holding the offset change, not the day after."""
    changes = []
    for i in range(gen.days):
        d = gen.day(i)
        begin = gen.local(d)
        end = gen.local(d + timedelta(days=1))
        hours = (end.astimezone(UTC) - begin.astimezone(UTC)).total_seconds() / 3600
        if hours != 24:
            changes.append(
                {
                    "date": d.isoformat(),
                    "utc_offset_minutes_before": int(begin.utcoffset().total_seconds() // 60),
                    "utc_offset_minutes_after": int(end.utcoffset().total_seconds() // 60),
                    "local_day_hours": hours,
                }
            )
    return {"transitions": changes}


def _dump(obj: object) -> bytes:
    return (json.dumps(obj, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _dump_pretty(obj: object) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def generate(seed: int, tz_name: str, start: date, days: int, volume: bool = False) -> dict[str, bytes]:
    """All files as ``{relative posix path: bytes}``; the writer is a separate step so tests need no disk."""
    gen = Generator(seed, tz_name, start, days)
    names = SCENARIOS + ([VOLUME_SCENARIO] if volume else [])
    files: dict[str, bytes] = {}
    dst = dst_facts(gen)
    for name in names:
        description, payloads = getattr(gen, _BUILDERS[name])()
        order = []
        for n, p in enumerate(payloads, 1):
            fname = f"{n:02d}.json"
            files[f"{name}/{fname}"] = _dump(p)
            order.append(fname)
        exp = {
            "format": 1,
            "scenario": name,
            "description": description,
            "seed": seed,
            "timezone": tz_name,
            "app_version": APP_VERSION,
            "enabled_types": ENABLED_TYPES,
            "delivery_order": order,
        }
        exp.update(expected_outcome(payloads))
        if name == "baseline_30_days":
            exp["dst"] = dict(dst)
            for change in exp["dst"]["transitions"]:
                night = next(
                    p["sleep"][0]["duration_seconds"] for p in payloads if p["daily_totals"][0]["date"] == change["date"]
                )
                change["sleep_seconds_across_transition"] = night
        files[f"{name}/expectations.json"] = _dump_pretty(exp)
    files["index.json"] = _dump_pretty(
        {"format": 1, "seed": seed, "timezone": tz_name, "app_version": APP_VERSION, "scenarios": names}
    )
    return files


def write(files: dict[str, bytes], out: Path) -> None:
    for rel, data in files.items():
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)  # bytes, not text mode, so Windows does not rewrite newlines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, help="directory to write the fixture set into")
    ap.add_argument("--seed", type=int, default=20251102)
    ap.add_argument("--tz", default=DEFAULT_TZ, help="IANA zone the synthetic phone is in")
    ap.add_argument("--start-date", default=DEFAULT_START, help="first local day, YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--volume", action="store_true", help="also write the 1 Hz heart-rate day (about 87 payloads)")
    ap.add_argument("--sign", type=Path, metavar="FILE", help="print the X-Signature value for FILE and exit")
    ap.add_argument("--secret", help="signing secret for --sign")
    args = ap.parse_args(argv)

    if args.sign:
        if args.secret is None:
            ap.error("--sign needs --secret")
        print(sign_body(args.secret, args.sign.read_bytes()))
        return 0
    if args.out is None:
        ap.error("--out is required")
    files = generate(args.seed, args.tz, date.fromisoformat(args.start_date), args.days, args.volume)
    write(files, args.out)
    print(f"wrote {len(files)} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
