from __future__ import annotations

import json
import pickle
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path

from seatflow.data.contracts import DailyBar, parse_trade_date
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.datasets.windows import WindowSample, build_window_samples
from seatflow.labels import (
    FROZEN_ADJUSTMENT_POLICY,
    FROZEN_CALENDAR_POLICY,
    RESEARCH_HORIZONS,
    StockLabel,
    generate_future_return_labels,
)

_REQUIRED_DATASETS = ("daily_bar", "dragon_tiger_seat_event")


@dataclass(frozen=True)
class VerifiedHistoricalSnapshots:
    """Immutable, manifest-verified inputs for real historical graph research."""

    provider: str
    trading_calendar: tuple[date, ...]
    bars: tuple[DailyBar, ...]
    seat_events: tuple[DragonTigerRecord, ...]
    manifest_hashes: Mapping[str, str]
    source_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "trading_calendar": [item.isoformat() for item in self.trading_calendar],
            "bar_count": len(self.bars),
            "seat_event_count": len(self.seat_events),
            "manifest_count": len(self.manifest_hashes),
            "manifest_hashes": dict(sorted(self.manifest_hashes.items())),
            "source_hash": self.source_hash,
        }


class MaturedLabelDiskCache(MutableMapping[tuple[date, str], tuple[StockLabel, ...]]):
    """SQLite-backed label cache that keeps long historical runs bounded in RAM."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS labels (cache_key TEXT PRIMARY KEY, payload BLOB NOT NULL)"
        )
        self._connection.commit()
        self._pending_writes = 0

    def __getitem__(self, key: tuple[date, str]) -> tuple[StockLabel, ...]:
        row = self._connection.execute(
            "SELECT payload FROM labels WHERE cache_key = ?", (_cache_key(key),)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        value = pickle.loads(bytes(row[0]))
        if not isinstance(value, tuple) or any(not isinstance(item, StockLabel) for item in value):
            raise ValueError("matured label disk cache contains an invalid payload")
        return value

    def __setitem__(self, key: tuple[date, str], value: tuple[StockLabel, ...]) -> None:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        self._connection.execute(
            "INSERT OR REPLACE INTO labels(cache_key, payload) VALUES (?, ?)",
            (_cache_key(key), payload),
        )
        self._pending_writes += 1
        if self._pending_writes >= 1000:
            self._connection.commit()
            self._pending_writes = 0

    def __delitem__(self, key: tuple[date, str]) -> None:
        cursor = self._connection.execute(
            "DELETE FROM labels WHERE cache_key = ?", (_cache_key(key),)
        )
        self._connection.commit()
        if cursor.rowcount == 0:
            raise KeyError(key)

    def __iter__(self) -> Iterator[tuple[date, str]]:
        rows = self._connection.execute("SELECT cache_key FROM labels")
        for (value,) in rows:
            trade_date, stock_id = str(value).split("|", maxsplit=1)
            yield date.fromisoformat(trade_date), stock_id

    def __len__(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM labels").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._connection.commit()
        self._connection.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _cache_key(key: tuple[date, str]) -> str:
    return f"{key[0].isoformat()}|{key[1]}"


def load_verified_historical_snapshots(
    data_root: str | Path,
    *,
    provider: str,
    trading_calendar: Sequence[date] | None = None,
    industry_membership_root: str | Path | None = None,
) -> VerifiedHistoricalSnapshots:
    """Load only snapshots whose manifest, hash, and quality contract all verify.

    If ``industry_membership_root`` points at a backfilled sw2021 membership
    directory (layout: ``<root>/<provider>/sw2021_l1_industry_membership/
    l1_code=*/is_new={Y,N}/data.json``), seat events with an empty
    ``industry`` field are filled in point-in-time: a stock's industry on
    ``trade_date`` is its ``l1_code`` whose membership interval
    ``in_date <= trade_date < out_date`` contains the trade date. This is
    the only data-side wiring needed to make industry relations carry
    signal in the graph builder; without it every seat event has
    ``industry=None`` and the graph has zero industry nodes.

    The parameter is opt-in so callers that reproduce the R5 evidence
    (where this wiring did not exist) keep their byte-identical inputs.
    """

    root = Path(data_root) / "processed" / provider
    if not root.exists():
        raise ValueError(f"processed snapshot root does not exist: {root}")
    available_dates = _snapshot_dates(root / "daily_bar")
    requested_calendar = tuple(trading_calendar) if trading_calendar is not None else None
    calendar = tuple(sorted(set(requested_calendar or available_dates)))
    if not calendar:
        raise ValueError("verified historical snapshots require a non-empty trading calendar")
    if requested_calendar is not None and requested_calendar != calendar:
        raise ValueError("trading calendar must be sorted and unique")

    bars: list[DailyBar] = []
    seat_events: list[DragonTigerRecord] = []
    manifest_hashes: dict[str, str] = {}
    for trade_date in calendar:
        for dataset in _REQUIRED_DATASETS:
            rows, digest = _read_verified_snapshot(root, provider, dataset, trade_date)
            manifest_hashes[f"{dataset}/{trade_date.isoformat()}"] = digest
            if dataset == "daily_bar":
                parsed_bars = tuple(DailyBar.from_dict(row) for row in rows)
                if any(bar.trade_date != trade_date for bar in parsed_bars):
                    raise ValueError(f"daily_bar rows have an unexpected trade date: {trade_date}")
                bars.extend(parsed_bars)
            else:
                parsed_events = tuple(_seat_event_from_dict(row) for row in rows)
                if any(record.trade_date != trade_date for record in parsed_events):
                    raise ValueError(
                        f"dragon_tiger_seat_event rows have an unexpected trade date: {trade_date}"
                    )
                seat_events.extend(parsed_events)
    if industry_membership_root is not None:
        membership_map = load_industry_membership_map(
            industry_membership_root, provider=provider
        )
        seat_events = [
            _fill_industry(record, membership_map) for record in seat_events
        ]
    source_hash = _hash_manifest_hashes(manifest_hashes)
    return VerifiedHistoricalSnapshots(
        provider=provider,
        trading_calendar=calendar,
        bars=tuple(bars),
        seat_events=tuple(seat_events),
        manifest_hashes=manifest_hashes,
        source_hash=source_hash,
    )


def build_real_window_samples(
    snapshots: VerifiedHistoricalSnapshots,
    *,
    target_stock_ids_by_date: Mapping[date, Sequence[str]],
    cutoff_for_date: Callable[[date], datetime],
    observed_label_end: date,
    decay_lambda: float,
    max_edge_age_days: int,
    cooccurrence_max_records_per_stock: int = 20,
    matured_label_cache: MutableMapping[tuple[date, str], tuple[StockLabel, ...]] | None = None,
) -> tuple[WindowSample, ...]:
    """Build 1/3/5-day windows without allowing labels beyond the released period."""

    if not target_stock_ids_by_date:
        raise ValueError("real window construction requires target stocks by as-of date")
    if observed_label_end not in snapshots.trading_calendar:
        raise ValueError("observed_label_end must be in the frozen trading calendar")
    normalized_targets: dict[date, tuple[str, ...]] = {}
    for as_of_date in sorted(target_stock_ids_by_date):
        if as_of_date not in snapshots.trading_calendar:
            raise ValueError(f"as_of_date is absent from the frozen calendar: {as_of_date}")
        target_stock_ids = tuple(sorted(set(target_stock_ids_by_date[as_of_date])))
        if not target_stock_ids:
            continue
        normalized_targets[as_of_date] = target_stock_ids
    matured_labels = _build_matured_labels(
        snapshots,
        cutoff_for_date=cutoff_for_date,
        observed_label_end=observed_label_end,
        cache=matured_label_cache,
    )
    built_samples = build_window_samples(
        snapshots.seat_events,
        snapshots.bars,
        as_of_dates=tuple(sorted(normalized_targets)),
        target_stock_ids=(),
        horizons=RESEARCH_HORIZONS,
        cutoff_for_date=cutoff_for_date,
        decay_lambda=decay_lambda,
        max_edge_age_days=max_edge_age_days,
        adjustment_policy=FROZEN_ADJUSTMENT_POLICY,
        calendar_policy=FROZEN_CALENDAR_POLICY,
        trading_calendar=snapshots.trading_calendar,
        matured_labels=matured_labels,
        target_stock_ids_by_date=normalized_targets,
        correlation_stock_ids_by_date=normalized_targets,
        graph_stock_ids_by_date=normalized_targets,
        cooccurrence_max_records_per_stock=cooccurrence_max_records_per_stock,
    )
    all_samples = tuple(
        sample
        for sample in built_samples
        if tuple(sorted(label.horizon for label in sample.labels)) == RESEARCH_HORIZONS
        and all(label.label_end_date <= observed_label_end for label in sample.labels)
    )
    if not all_samples:
        raise ValueError("no fully observed 1/3/5-day real graph windows were produced")
    return all_samples


def _build_matured_labels(
    snapshots: VerifiedHistoricalSnapshots,
    *,
    cutoff_for_date: Callable[[date], datetime],
    observed_label_end: date,
    cache: MutableMapping[tuple[date, str], tuple[StockLabel, ...]] | None = None,
) -> tuple[StockLabel, ...]:
    bars_by_stock: dict[str, list[DailyBar]] = defaultdict(list)
    for bar in snapshots.bars:
        bars_by_stock[bar.stock_id].append(bar)
    event_keys = tuple(
        sorted({(record.trade_date, record.stock_id) for record in snapshots.seat_events})
    )
    labels: list[StockLabel] = []
    for as_of_date, stock_id in event_keys:
        cache_key = (as_of_date, stock_id)
        if cache is not None and cache_key in cache:
            labels.extend(cache[cache_key])
            continue
        stock_bars = bars_by_stock.get(stock_id)
        if not stock_bars:
            if cache is not None:
                cache[cache_key] = ()
            continue
        try:
            generated = generate_future_return_labels(
                stock_bars,
                as_of_date=as_of_date,
                cutoff_time=cutoff_for_date(as_of_date),
                target_stock_ids=(stock_id,),
                horizons=RESEARCH_HORIZONS,
                adjustment_policy=FROZEN_ADJUSTMENT_POLICY,
                calendar_policy=FROZEN_CALENDAR_POLICY,
                trading_calendar=snapshots.trading_calendar,
            )
        except ValueError as error:
            if "no_as_of bar" not in str(error) and "no as_of bar" not in str(error):
                raise
            if cache is not None:
                cache[cache_key] = ()
            continue
        filtered = tuple(
            label for label in generated if label.label_end_date <= observed_label_end
        )
        if cache is not None:
            cache[cache_key] = filtered
        labels.extend(filtered)
    return tuple(labels)


def _snapshot_dates(dataset_root: Path) -> tuple[date, ...]:
    if not dataset_root.exists():
        raise ValueError(f"required snapshot dataset does not exist: {dataset_root}")
    dates: list[date] = []
    for child in dataset_root.glob("date=*"):
        if not child.is_dir():
            continue
        try:
            dates.append(date.fromisoformat(child.name.removeprefix("date=")))
        except ValueError as error:
            raise ValueError(f"invalid snapshot date directory: {child}") from error
    return tuple(sorted(set(dates)))


def _read_verified_snapshot(
    root: Path,
    provider: str,
    dataset: str,
    trade_date: date,
) -> tuple[tuple[dict[str, object], ...], str]:
    snapshot_root = root / dataset / f"date={trade_date.isoformat()}"
    manifest_path = snapshot_root / "manifest.json"
    data_path = snapshot_root / "data.json"
    if not manifest_path.exists() or not data_path.exists():
        raise ValueError(f"required snapshot is missing: {dataset}/{trade_date.isoformat()}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_rows = json.loads(data_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"snapshot is unreadable: {dataset}/{trade_date.isoformat()}") from error
    if not isinstance(manifest, dict) or not isinstance(raw_rows, list):
        raise ValueError(
            f"snapshot payload has an invalid shape: {dataset}/{trade_date.isoformat()}"
        )
    rows = tuple(_require_object(row, dataset) for row in raw_rows)
    digest = sha256(data_path.read_bytes()).hexdigest()
    metadata = manifest.get("metadata")
    quality = metadata.get("quality") if isinstance(metadata, dict) else None
    if (
        manifest.get("provider") != provider
        or manifest.get("dataset") != dataset
        or manifest.get("trade_date") != trade_date.isoformat()
        or manifest.get("sha256") != digest
        or manifest.get("row_count") != len(rows)
        or not isinstance(quality, dict)
        or quality.get("passed") is not True
    ):
        raise ValueError(
            f"snapshot manifest verification failed: {dataset}/{trade_date.isoformat()}"
        )
    return rows, digest


def _seat_event_from_dict(row: Mapping[str, object]) -> DragonTigerRecord:
    available_time = row.get("available_time")
    return DragonTigerRecord(
        trade_date=parse_trade_date(row["trade_date"]),
        seat_id=str(row["seat_id"]),
        stock_id=str(row["stock_id"]),
        side=TradeSide(str(row["side"])),
        amount=float(str(row["amount"])),
        industry=_optional_text(row.get("industry")),
        concepts=_string_tuple(row.get("concepts")),
        event_reason=_optional_text(row.get("event_reason")),
        available_time=datetime.fromisoformat(str(available_time)) if available_time else None,
        source_name=str(row.get("source_name") or "unknown"),
        source_version=_optional_text(row.get("source_version")),
        source_row_count=int(str(row.get("source_row_count", 1))),
        source_side_set=_string_tuple(row.get("source_side_set")),
        identity_scope=str(row.get("identity_scope") or "named_seat"),
    )


def load_industry_membership_map(
    root: str | Path,
    *,
    provider: str,
) -> dict[date, dict[str, str]]:
    """Load the sw2021 L1 industry membership as a point-in-time map.

    Walks ``<root>/<provider>/sw2021_l1_industry_membership/l1_code=*/
    is_new={Y,N}/data.json`` and returns ``{trade_date: {stock_id:
    l1_code}}`` where every ``trade_date`` is the first day a stock is a
    member of an L1 industry and the membership ends on the day before
    ``out_date`` (None means still active). Each trade_date is one calendar
    day; downstream code should fall back to the most recent prior entry
    for any trade_date not in the map. Empty / missing root is a hard
    error — the wiring is opt-in but it must be consistent.
    """
    membership_root = Path(root) / provider / "sw2021_l1_industry_membership"
    if not membership_root.exists():
        raise ValueError(f"industry membership root does not exist: {membership_root}")
    intervals: list[tuple[date, date | None, str, str]] = []
    for l1_dir in sorted(membership_root.glob("l1_code=*")):
        if not l1_dir.is_dir():
            continue
        l1_code = l1_dir.name.removeprefix("l1_code=")
        for is_new_dir in sorted(l1_dir.glob("is_new=*")):
            if not is_new_dir.is_dir():
                continue
            data_path = is_new_dir / "data.json"
            if not data_path.exists():
                continue
            try:
                rows = json.loads(data_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"industry membership file is unreadable: {data_path}"
                ) from error
            if not isinstance(rows, list):
                raise ValueError(
                    f"industry membership file must be a list: {data_path}"
                )
            for row in rows:
                if not isinstance(row, dict):
                    continue
                stock_id = row.get("stock_id")
                if not isinstance(stock_id, str) or not stock_id.strip():
                    continue
                in_date_raw = row.get("in_date")
                if not isinstance(in_date_raw, str) or not in_date_raw:
                    continue
                try:
                    in_date = date.fromisoformat(in_date_raw)
                except ValueError:
                    continue
                out_date_raw = row.get("out_date")
                out_date: date | None = None
                if isinstance(out_date_raw, str) and out_date_raw:
                    try:
                        out_date = date.fromisoformat(out_date_raw)
                    except ValueError:
                        out_date = None
                intervals.append((in_date, out_date, stock_id, l1_code))

    by_day: dict[date, dict[str, str]] = defaultdict(dict)
    for in_date, out_date, stock_id, l1_code in intervals:
        if in_date not in by_day[in_date]:
            by_day[in_date][stock_id] = l1_code
        if out_date is not None and out_date not in by_day[out_date]:
            by_day[out_date][stock_id] = "__LEAVE__"
    return dict(sorted(by_day.items()))


def _fill_industry(
    record: DragonTigerRecord,
    membership_by_day: Mapping[date, Mapping[str, str]],
) -> DragonTigerRecord:
    """Return ``record`` with ``industry`` set from PIT membership if empty.

    Membership changes on two kinds of days: ``in_date`` (stock joins an
    L1) and ``out_date`` (stock leaves). We walk the day-sorted membership
    keys and track the most recent l1_code for each stock; a stock not seen
    since the start has no industry. The ``__LEAVE__`` sentinel clears the
    tracked l1_code on out_date.
    """
    if record.industry:
        return record
    tracked: dict[str, str] = {}
    for day, snapshot in membership_by_day.items():
        if day > record.trade_date:
            break
        for stock_id, value in snapshot.items():
            if value == "__LEAVE__":
                tracked.pop(stock_id, None)
            else:
                tracked[stock_id] = value
    l1_code = tracked.get(record.stock_id)
    if l1_code is None:
        return record
    return replace(record, industry=l1_code)


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _string_tuple(value: object | None) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _require_object(value: object, dataset: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{dataset} snapshot contains a non-object row")
    return value


def _hash_manifest_hashes(manifest_hashes: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(manifest_hashes.items())), separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()
