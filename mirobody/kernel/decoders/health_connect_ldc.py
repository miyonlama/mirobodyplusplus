"""Life Dashboard Companion webhook payloads → facts. Pure; field names are theirs.

[Life Dashboard Companion](https://github.com/owen282000/life-dashboard-companion-app)
reads Health Connect on an Android phone and POSTs JSON to a receiver. This
module is the decode half of that receiver: one payload dict in, standardised
rows and counted refusals out. Signature checks, storage and retries belong to
whoever runs the endpoint.

## What needs care

* **Keys are LDC's, classes are androidx's.** `res/crosswalks/health_connect_ldc.tsv`
  maps each documented key to the Record class in `health_connect.tsv` and to a
  catalogue metric, or declines it WITH A REASON. A key the table does not know
  is refused as `unmapped-key`, never guessed into a neighbouring metric.
* **The unit is the catalogue's.** The table states the unit LDC sends; the
  target is `metrics.METRICS[metric].unit_ucum`, and the conversion is
  `mirobody.units`. A pair that does not convert is refused, not relabelled.
* **One heart rate sample is one object.** Heart rate, speed and skin temperature
  go out as `<record id>#<epoch ms>`; the whole string is kept as the vendor
  record id so two samples never share an identity. A deleted record names the
  bare id, which is what the `is_prefix` flag of `deletions` is for.
* **Times are UTC; days are not.** An instant is exact. The `daily_totals` days
  are cut in the payload's own IANA zone (the caller's zone is the fallback), so
  the day of a DST change is 23 or 25 hours long.
* **iOS shares the key, not the number.** `healthkit_ios` sends SDNN under
  `heart_rate_variability` where Android sends RMSSD; such a payload is refused
  whole as `wrong-source` rather than filed under `hrvRMSSD`.
* **Bucketed series are not readings.** A key named in `_resolutions` carries
  averages, not measurements; it is refused as `bucketed-unsupported`.
* **Reproductive data is opt-in.** Cycle keys are refused as `type-not-enabled`
  unless the caller passes `accept_reproductive=True`.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from importlib import resources

from mirobody import units
from mirobody.kernel import metrics
from mirobody.kernel.series import Fact, day_bounds_ms, zone
from ._common import number

#: This decoder reads one thing: a whole webhook payload.
DATA_TYPES: tuple[str, ...] = ("payload",)

#: The `source` of a payload this decoder reads. `healthkit_ios` and
#: `screen_time` share the envelope, not the meaning of the fields.
SOURCE = "health_connect"

#: Keys whose objects are samples of a longer record, sent with
#: `<record id>#<epoch ms>` as `uuid`. A sleep session's stage rows use the same
#: spelling, so a deletion has the same prefix rule for it.
SAMPLE_KEYS = frozenset({"heart_rate", "speed", "skin_temperature"})
PREFIX_KEYS = SAMPLE_KEYS | {"sleep"}

#: Refusal codes, in the order they are decided.
REASONS: tuple[str, ...] = (
    "wrong-source",
    "bucketed-unsupported",
    "unmapped-key",
    "type-not-enabled",
    "no-metric",
    "unit-mismatch",
    "malformed",
)

#: Top-level keys that are envelope, not records.
ENVELOPE = frozenset(
    {
        "timestamp", "timezone", "app_version", "source", "sequence", "daily_totals", "_resolutions",
        "deleted_records", "deletions_unavailable", "records_outside_window", "backfill", "window_start",
        "window_end", "window_complete", "writeback", "_diagnostics", "test", "message", "device",
        "app_filter", "screen_time",
    }
)


@dataclass(frozen=True)
class Crosswalk:
    """One row of the LDC table."""

    ldc_key: str
    ldc_field: str
    hc_class: str
    hc_field: str
    metric: str
    src_unit: str
    time: str
    gate: str
    reason: str
    note: str

    @property
    def declined(self) -> bool:
        return not self.metric

    @property
    def vendor_field(self) -> str:
        return self.hc_class


def _load_crosswalk() -> dict[str, list[Crosswalk]]:
    text = resources.files("mirobody").joinpath("res", "crosswalks", "health_connect_ldc.tsv").read_text(encoding="utf-8")
    body = "".join(line for line in io.StringIO(text) if not line.startswith("#"))
    out: dict[str, list[Crosswalk]] = {}
    for row in csv.DictReader(io.StringIO(body), delimiter="\t", quoting=csv.QUOTE_NONE):
        c = Crosswalk(*(row[k] or "" for k in (
            "ldc_key", "ldc_field", "hc_class", "hc_field", "metric", "src_unit", "time", "gate", "reason", "note")))
        out.setdefault(c.ldc_key, []).append(c)
    return out


CROSSWALK: dict[str, list[Crosswalk]] = _load_crosswalk()

#: Every catalogue metric this decoder can emit, read from its own table.
METRICS: frozenset[str] = frozenset(c.metric for rows in CROSSWALK.values() for c in rows if c.metric)


@dataclass(frozen=True)
class Row:
    """A decoded fact with what `Fact` has no field for."""

    fact: Fact
    key: str
    vendor_field: str
    record_uuid: str


@dataclass(frozen=True)
class Decoded:
    rows: tuple[Row, ...]
    #: `(key, reason, count)`, sorted. A key may appear once per reason.
    refusals: tuple[tuple[str, str, int], ...]


def _to_catalogue_unit(value: float, src_unit: str, metric: str) -> float | None:
    """`value` in the catalogue's unit for `metric`, or `None` when the pair does
    not convert. The VO2 max trap lives here: LDC says mL/kg/min, the catalogue
    mL/min/kg, and the engine normalises both to one spelling."""
    row = metrics.METRICS.get(metric)
    target = (row.unit_ucum if row else "") or ""
    source = units.normalize_unit(src_unit) or src_unit
    if not target or source == target:
        return value if target else None
    if units.convertible(source, target):
        return units.convert_value(value, source, target)
    return None


#: Fraction digits beyond microseconds, which `fromisoformat` does not take on every version.
_FRACTION = re.compile(r"(\.[0-9]{6})[0-9]+")


def _ms(text: object) -> int:
    """An ISO 8601 instant with `Z` or an offset → unix ms; `0` otherwise.

    LDC always sends UTC, with 0, 3, 6 or 9 fraction digits. A string with no
    offset is refused rather than read as UTC: this payload never has one, so
    its absence means something else produced it.
    """
    if not isinstance(text, str) or "T" not in text:
        return 0
    s = text.strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    s = _FRACTION.sub(lambda m: m.group(1), s)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return 0
    if dt.tzinfo is None:
        return 0
    ms = int(round((dt - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() * 1000))
    return ms if ms > 0 else 0


def _payload_zone(payload: dict, fallback: str) -> str:
    """The payload's own IANA zone, else the caller's. An unknown name would
    fall back to UTC inside `zone`, which cuts every day at the wrong hour."""
    name = payload.get("timezone")
    if isinstance(name, str) and name:
        try:
            zone(name, strict=True)
        except ValueError:
            return fallback
        return name
    return fallback


def _series_key(rec: dict) -> str:
    return "hc:" + str(rec.get("source") or "")


def _device_id(rec: dict) -> str:
    meta = rec.get("metadata")
    dev = meta.get("device") if isinstance(meta, dict) else None
    return str(dev.get("model") or "") if isinstance(dev, dict) else ""


class _Run:
    """One payload's decode. Rows and refusals accumulate here."""

    def __init__(self, payload: dict, tz: str, accept_reproductive: bool, ingested_at_ms: int) -> None:
        self.payload = payload
        self.tz = _payload_zone(payload, tz)
        self.accept_reproductive = accept_reproductive
        self.ingested = ingested_at_ms
        self.rows: list[Row] = []
        self.refused: dict[tuple[str, str], int] = {}

    def refuse(self, key: str, reason: str, n: int = 1) -> None:
        if n > 0:
            self.refused[(key, reason)] = self.refused.get((key, reason), 0) + n

    def emit(self, c: Crosswalk, rec: dict, value: float, start: int, end: int, uuid: str, *,
             statistic: str = "", panel_id: str = "", suffix_id: str = "") -> None:
        converted = _to_catalogue_unit(value, c.src_unit, c.metric)
        label = c.ldc_key + (f".{c.ldc_field}" if c.ldc_field else "")
        if converted is None or not math.isfinite(converted):
            self.refuse(label, "unit-mismatch")
            return
        self.rows.append(
            Row(
                fact=Fact(
                    metric_key=c.metric,
                    value_num=converted,
                    effective_start_ms=start,
                    effective_end_ms=end if end > start else 0,
                    unit=metrics.METRICS[c.metric].unit_ucum,
                    series_key=_series_key(rec),
                    device_id=_device_id(rec),
                    panel_id=panel_id,
                    statistic=statistic,
                    modality="sensed",
                    source_record_id=suffix_id or uuid,
                    ingested_at_ms=self.ingested,
                ),
                key=c.ldc_key,
                vendor_field=c.vendor_field,
                record_uuid=uuid,
            )
        )

    # -- record arrays -----------------------------------------------------

    def key(self, key: str, records: list, resolutions: dict) -> None:
        table = CROSSWALK.get(key)
        if table is None:
            self.refuse(key, "unmapped-key", len(records))
            return
        if key in resolutions:
            self.refuse(key, "bucketed-unsupported", len(records))
            return
        if table[0].gate == "reproductive" and not self.accept_reproductive:
            self.refuse(key, "type-not-enabled", len(records))
            return
        if all(c.declined for c in table):
            self.refuse(key, "no-metric", len(records))
            return
        for rec in records:
            if not isinstance(rec, dict):
                self.refuse(key, "malformed")
            elif "bucket_start" in rec:
                self.refuse(key, "bucketed-unsupported")
            elif key == "sleep":
                self.sleep(table, rec)
            else:
                self.record(key, table, rec)

    def record(self, key: str, table: list[Crosswalk], rec: dict) -> None:
        uuid = rec.get("uuid")
        if not isinstance(uuid, str) or not uuid:
            self.refuse(key, "malformed")
            return
        t = table[0].time
        if t == "instant":
            start = end = _ms(rec.get("time"))
        else:
            start, end = _ms(rec.get("start_time")), _ms(rec.get("end_time"))
            if end and end < start:
                start = 0
        if not start:
            self.refuse(key, "malformed")
            return
        panel = uuid if sum(1 for c in table if c.metric) > 1 else ""
        for c in table:
            if c.declined:
                if c.ldc_field in rec:
                    self.refuse(f"{key}.{c.ldc_field}", c.reason or "no-metric")
                continue
            value = number(rec.get(c.ldc_field))
            if value is None or not math.isfinite(value):
                self.refuse(f"{key}.{c.ldc_field}", "malformed")
                continue
            self.emit(c, rec, value, start, end, uuid, panel_id=panel)

    def sleep(self, table: list[Crosswalk], rec: dict) -> None:
        by_field = {c.ldc_field: c for c in table}
        uuid = rec.get("uuid")
        if not isinstance(uuid, str) or not uuid:
            self.refuse("sleep", "malformed")
            return
        end = _ms(rec.get("session_end_time"))
        dur = number(rec.get("duration_seconds"))
        if not end or dur is None or dur <= 0:
            self.refuse("sleep.session", "malformed")
        else:
            start = end - int(round(dur * 1000))
            self.emit(by_field["session"], rec, float(end - start), start, end, uuid)
        for stage in rec.get("stages") or []:
            kind = str(stage.get("stage") or "") if isinstance(stage, dict) else ""
            c = by_field.get(f"stages:{kind}")
            if c is None:
                self.refuse(f"sleep.stages:{kind or '?'}", "unmapped-key")
                continue
            if c.declined:
                self.refuse(f"sleep.{c.ldc_field}", c.reason or "no-metric")
                continue
            s, e = _ms(stage.get("start_time")), _ms(stage.get("end_time"))
            if not s or e <= s:
                self.refuse(f"sleep.{c.ldc_field}", "malformed")
                continue
            # Two light stages of one night share a record uuid; the start
            # keeps their identities apart.
            self.emit(c, rec, float(e - s), s, e, uuid, suffix_id=f"{uuid}#{s}")

    # -- daily_totals ------------------------------------------------------

    def daily_totals(self, entries: list) -> None:
        table = {c.ldc_field: c for c in CROSSWALK["daily_totals"]}
        for entry in entries:
            try:
                day = date.fromisoformat(str(entry.get("date"))) if isinstance(entry, dict) else None
            except ValueError:
                day = None
            if day is None:
                self.refuse("daily_totals", "malformed")
                continue
            start, end = day_bounds_ms(day, self.tz)
            for field, c in table.items():
                if field not in entry:
                    continue
                if c.declined:
                    self.refuse(f"daily_totals.{field}", c.reason or "no-metric")
                    continue
                value = number(entry.get(field))
                if value is None or not math.isfinite(value):
                    self.refuse(f"daily_totals.{field}", "malformed")
                    continue
                self.emit(c, entry, value, start, end, f"daily_totals:{day.isoformat()}:{field}", statistic="provider_daily")

    # -- whole payload -----------------------------------------------------

    def run(self) -> Decoded:
        p = self.payload
        records = {k: v for k, v in p.items() if k not in ENVELOPE and isinstance(v, list)}
        if p.get("test"):
            return Decoded((), ())
        if str(p.get("source") or "") != SOURCE:
            for k, v in records.items():
                self.refuse(k, "wrong-source", len(v))
            if isinstance(p.get("daily_totals"), list):
                self.refuse("daily_totals", "wrong-source", len(p["daily_totals"]))
        else:
            res = p.get("_resolutions")
            resolutions = res if isinstance(res, dict) else {}
            for k, v in records.items():
                self.key(k, v, resolutions)
            if isinstance(p.get("daily_totals"), list):
                self.daily_totals(p["daily_totals"])
        refusals = tuple(sorted((k, r, n) for (k, r), n in self.refused.items()))
        return Decoded(tuple(self.rows), refusals)


def decode_payload(payload: dict, tz: str, *, accept_reproductive: bool = False, ingested_at_ms: int = 0) -> Decoded:
    """Rows and refusals from one payload, decided once so they cannot disagree."""
    if not isinstance(payload, dict):
        return Decoded((), ())
    return _Run(payload, tz, accept_reproductive, ingested_at_ms).run()


def decode_rows(payload: dict, tz: str, *, accept_reproductive: bool = False, ingested_at_ms: int = 0) -> list[Row]:
    """Decoded rows: the fact, the LDC key, the Health Connect class as the
    vendor field, and the record uuid."""
    return list(decode_payload(payload, tz, accept_reproductive=accept_reproductive, ingested_at_ms=ingested_at_ms).rows)


def refusals(payload: dict, tz: str = "UTC", *, accept_reproductive: bool = False) -> list[tuple[str, str, int]]:
    """What the payload carried that was NOT decoded, as `(key, reason, count)`
    with `reason` one of `REASONS`. Published so a gap is counted, not silent."""
    return list(decode_payload(payload, tz, accept_reproductive=accept_reproductive).refusals)


def decode(
    data_type: str,
    item: dict,
    tz: str,
    *,
    pulled_at_ms: int = 0,
    source_record_id: str = "",
    ingested_at_ms: int = 0,
    accept_reproductive: bool = False,
) -> list[Fact]:
    """The `decoders.decode` contract: facts from one payload; `[]` for any other
    shape. `source_record_id` is unused, each row carries its own record uuid."""
    if data_type != "payload":
        return []
    return [r.fact for r in decode_rows(item, tz, accept_reproductive=accept_reproductive,
                                        ingested_at_ms=ingested_at_ms or pulled_at_ms)]


def deletions(payload: dict) -> list[tuple[str, str, bool]]:
    """`(key, uuid, is_prefix)` for every `deleted_records` entry of a type this
    decoder maps.

    The consumer retracts rows of `key` whose vendor record id equals `uuid`;
    with `is_prefix` it also retracts ids starting with `uuid + "#"`, which is
    how samples and sleep stages are named. A deleted uuid can arrive again as
    a record later (Fitbit rewrites sleep that way), so a retraction is not a
    tombstone for the id.
    """
    out: list[tuple[str, str, bool]] = []
    seen: set[tuple[str, str]] = set()
    entries = payload.get("deleted_records") if isinstance(payload, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        key, uuid = entry.get("type"), entry.get("uuid")
        if not isinstance(key, str) or not isinstance(uuid, str) or not uuid:
            continue
        table = CROSSWALK.get(key)
        if table is None or all(c.declined for c in table) or (key, uuid) in seen:
            continue
        seen.add((key, uuid))
        out.append((key, uuid, key in PREFIX_KEYS))
    return out


def declined_keys() -> dict[str, str]:
    """`key[.field]` → reason for everything the table declines, so a consumer
    sees the gaps instead of discovering them as silence."""
    return {
        c.ldc_key + (f".{c.ldc_field}" if c.ldc_field else ""): c.note
        for rows in CROSSWALK.values()
        for c in rows
        if c.declined
    }


__all__ = [
    "CROSSWALK",
    "Crosswalk",
    "DATA_TYPES",
    "Decoded",
    "METRICS",
    "REASONS",
    "Row",
    "decode",
    "decode_payload",
    "decode_rows",
    "declined_keys",
    "deletions",
    "refusals",
]
