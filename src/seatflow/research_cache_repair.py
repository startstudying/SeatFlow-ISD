"""Fail-closed repair of historical research daily-bar caches."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

from seatflow.data.contracts import DailyBar, normalize_stock_id, parse_trade_date
from seatflow.data.providers import ProviderError


class DailyHistoryProvider(Protocol):
    """The narrow provider surface required to repair one stock history."""

    name: str

    def fetch_daily_bar_history(
        self, stock_id: str, start_date: date, end_date: date
    ) -> tuple[DailyBar, ...]: ...


@dataclass(frozen=True)
class CacheRepairRequest:
    stock_id: str
    execution_dates: tuple[date, ...]
    cache_path: Path


@dataclass(frozen=True)
class StockCacheRepairResult:
    stock_id: str
    cache_path: Path
    requested_execution_dates: tuple[date, ...]
    unresolved_execution_dates: tuple[date, ...]
    original_sha256: str | None
    updated_sha256: str | None
    fetched_sha256: str | None
    backup_path: Path | None
    backup_sha256: str | None
    added: int
    replaced: int
    fetched_rows: int
    error: str | None

    @property
    def passed(self) -> bool:
        return self.error is None and not self.unresolved_execution_dates

    def to_dict(self) -> dict[str, object]:
        return {
            "stock_id": self.stock_id,
            "cache_path": str(self.cache_path),
            "requested_execution_dates": [
                item.isoformat() for item in self.requested_execution_dates
            ],
            "unresolved_execution_dates": [
                item.isoformat() for item in self.unresolved_execution_dates
            ],
            "original_sha256": self.original_sha256,
            "updated_sha256": self.updated_sha256,
            "fetched_sha256": self.fetched_sha256,
            "backup_path": str(self.backup_path) if self.backup_path is not None else None,
            "backup_sha256": self.backup_sha256,
            "added": self.added,
            "replaced": self.replaced,
            "fetched_rows": self.fetched_rows,
            "error": self.error,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ResearchCacheRepairReport:
    provider: str
    data_root: Path
    start_date: date
    end_date: date
    input_record_count: int
    results: tuple[StockCacheRepairResult, ...]

    @property
    def unresolved(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "stock_id": result.stock_id,
                "execution_date": execution_date.isoformat(),
            }
            for result in self.results
            for execution_date in result.unresolved_execution_dates
        )

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "provider": self.provider,
            "data_root": str(self.data_root),
            "repair_range": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
            },
            "input_record_count": self.input_record_count,
            "unique_stock_count": len(self.results),
            "requested_execution_bar_count": sum(
                len(result.requested_execution_dates) for result in self.results
            ),
            "unresolved_execution_bar_count": len(self.unresolved),
            "unresolved": list(self.unresolved),
            "results": [result.to_dict() for result in self.results],
            "passed": self.passed,
        }


def repair_research_cache(
    provider: DailyHistoryProvider,
    data_root: Path,
    start_date: date,
    end_date: date,
    missing_execution_jsonl: Path,
) -> ResearchCacheRepairReport:
    """Refresh whole stock caches and prove every requested execution bar exists.

    Existing rows outside a provider response are preserved. This prevents a temporary
    provider gap from deleting previously recorded historical evidence while still
    replacing every date returned by the fresh full-range query.
    """

    if end_date < start_date:
        raise ValueError("research cache repair end date must not be before start date")
    requests, input_record_count = load_cache_repair_requests(
        missing_execution_jsonl, data_root, start_date, end_date
    )
    results = tuple(
        _repair_stock_cache(provider, request, start_date, end_date) for request in requests
    )
    return ResearchCacheRepairReport(
        provider=provider.name,
        data_root=data_root,
        start_date=start_date,
        end_date=end_date,
        input_record_count=input_record_count,
        results=results,
    )


def load_cache_repair_requests(
    missing_execution_jsonl: Path,
    data_root: Path,
    start_date: date,
    end_date: date,
) -> tuple[tuple[CacheRepairRequest, ...], int]:
    """Load and deduplicate the stable missing-execution JSONL contract."""

    try:
        lines = missing_execution_jsonl.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise ValueError(
            f"cannot read missing execution JSONL: {missing_execution_jsonl}"
        ) from error

    grouped_dates: dict[tuple[str, Path], set[date]] = {}
    paths_by_stock: dict[str, set[Path]] = {}
    input_record_count = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        input_record_count += 1
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL record at line {line_number}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"JSONL record at line {line_number} must be an object")
        stock_id = _required_stock_id(raw, line_number)
        execution_date = _required_execution_date(raw, line_number)
        cache_path = _cache_path_from_record(
            raw, data_root, stock_id, start_date, end_date, line_number
        )
        paths_by_stock.setdefault(stock_id, set()).add(cache_path)
        grouped_dates.setdefault((stock_id, cache_path), set()).add(execution_date)

    if input_record_count == 0:
        raise ValueError("missing execution JSONL contains no records")
    conflicting_stocks = sorted(
        stock_id for stock_id, paths in paths_by_stock.items() if len(paths) != 1
    )
    if conflicting_stocks:
        raise ValueError(
            "one stock maps to multiple cache paths: " + ", ".join(conflicting_stocks)
        )
    return (
        tuple(
            CacheRepairRequest(
                stock_id=stock_id,
                execution_dates=tuple(sorted(execution_dates)),
                cache_path=cache_path,
            )
            for (stock_id, cache_path), execution_dates in sorted(
                grouped_dates.items(), key=lambda item: item[0][0]
            )
        ),
        input_record_count,
    )


def write_research_cache_repair_report(
    output: Path, report: ResearchCacheRepairReport
) -> Path:
    """Persist the complete audit report atomically, even for negative results."""

    _atomic_write(output, _json_bytes(report.to_dict()))
    return output


def _repair_stock_cache(
    provider: DailyHistoryProvider,
    request: CacheRepairRequest,
    start_date: date,
    end_date: date,
) -> StockCacheRepairResult:
    original_sha256: str | None = None
    original_bytes: bytes | None = None
    try:
        if any(item < start_date or item > end_date for item in request.execution_dates):
            return _failed_result(
                request,
                "requested execution date is outside the full cache repair range",
            )
        cached_rows: dict[date, dict[str, object]] = {}
        if request.cache_path.exists():
            original_bytes = request.cache_path.read_bytes()
            original_sha256 = _sha256_bytes(original_bytes)
            cached_rows = _load_cached_rows(request.cache_path, request.stock_id)

        fetched_bars = provider.fetch_daily_bar_history(request.stock_id, start_date, end_date)
        fetched_rows = _normalized_fetched_rows(
            fetched_bars, request.stock_id, start_date, end_date
        )
        fetched_sha256 = _sha256_bytes(_json_bytes({"rows": list(fetched_rows.values())}))
        added = sum(item not in cached_rows for item in fetched_rows)
        replaced = sum(item in cached_rows for item in fetched_rows)
        merged_rows = dict(cached_rows)
        merged_rows.update(fetched_rows)
        unresolved = tuple(
            item for item in request.execution_dates if item not in merged_rows
        )

        # Refetch before making any destructive-looking change, then preserve the
        # original bytes as a hash-addressed recovery artifact.
        backup_path: Path | None = None
        backup_sha256: str | None = None
        if original_bytes is not None and original_sha256 is not None:
            current_sha256 = _sha256_file(request.cache_path)
            if current_sha256 != original_sha256:
                raise ValueError("cache changed while the repair query was in flight")
            backup_path = _backup_path(request.cache_path, original_sha256)
            _write_backup(backup_path, original_bytes, original_sha256)
            backup_sha256 = original_sha256

        updated_payload = {
            "rows": [merged_rows[item] for item in sorted(merged_rows)],
        }
        _atomic_write(request.cache_path, _gzip_json_bytes(updated_payload))
        updated_sha256 = _sha256_file(request.cache_path)
        return StockCacheRepairResult(
            stock_id=request.stock_id,
            cache_path=request.cache_path,
            requested_execution_dates=request.execution_dates,
            unresolved_execution_dates=unresolved,
            original_sha256=original_sha256,
            updated_sha256=updated_sha256,
            fetched_sha256=fetched_sha256,
            backup_path=backup_path,
            backup_sha256=backup_sha256,
            added=added,
            replaced=replaced,
            fetched_rows=len(fetched_rows),
            error=None,
        )
    except (OSError, ProviderError, ValueError) as error:
        return _failed_result(request, str(error), original_sha256=original_sha256)


def _failed_result(
    request: CacheRepairRequest,
    error: str,
    *,
    original_sha256: str | None = None,
) -> StockCacheRepairResult:
    return StockCacheRepairResult(
        stock_id=request.stock_id,
        cache_path=request.cache_path,
        requested_execution_dates=request.execution_dates,
        unresolved_execution_dates=request.execution_dates,
        original_sha256=original_sha256,
        updated_sha256=None,
        fetched_sha256=None,
        backup_path=None,
        backup_sha256=None,
        added=0,
        replaced=0,
        fetched_rows=0,
        error=error,
    )


def _load_cached_rows(path: Path, stock_id: str) -> dict[date, dict[str, object]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid research cache: {path}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError(f"research cache rows must be a JSON list: {path}")
    result: dict[date, dict[str, object]] = {}
    for raw in payload["rows"]:
        if not isinstance(raw, dict):
            raise ValueError(f"research cache row must be an object: {path}")
        bar = DailyBar.from_dict(raw)
        if bar.stock_id != stock_id:
            raise ValueError(f"research cache contains a different stock: {path}")
        if bar.trade_date in result:
            raise ValueError(f"research cache contains duplicate trade dates: {path}")
        result[bar.trade_date] = bar.to_dict()
    return result


def _normalized_fetched_rows(
    bars: Iterable[DailyBar],
    stock_id: str,
    start_date: date,
    end_date: date,
) -> dict[date, dict[str, object]]:
    result: dict[date, dict[str, object]] = {}
    for bar in bars:
        if bar.stock_id != stock_id:
            raise ValueError(f"provider returned a different stock for {stock_id}")
        if bar.trade_date < start_date or bar.trade_date > end_date:
            raise ValueError(
                f"provider returned a bar outside the full cache repair range for {stock_id}"
            )
        if bar.trade_date in result:
            raise ValueError(f"provider returned duplicate trade dates for {stock_id}")
        result[bar.trade_date] = bar.to_dict()
    return result


def _required_stock_id(raw: dict[object, object], line_number: int) -> str:
    value = raw.get("stock_id")
    if value is None or not str(value).strip():
        raise ValueError(f"JSONL record at line {line_number} requires stock_id")
    try:
        return normalize_stock_id(value)
    except ValueError as error:
        raise ValueError(f"invalid stock_id at line {line_number}") from error


def _required_execution_date(raw: dict[object, object], line_number: int) -> date:
    value = raw.get("execution_date")
    if value is None or not str(value).strip():
        raise ValueError(f"JSONL record at line {line_number} requires execution_date")
    try:
        return parse_trade_date(value)
    except ValueError as error:
        raise ValueError(f"invalid execution_date at line {line_number}") from error


def _cache_path_from_record(
    raw: dict[object, object],
    data_root: Path,
    stock_id: str,
    start_date: date,
    end_date: date,
    line_number: int,
) -> Path:
    raw_path = raw.get("cache_path")
    if raw_path is None or not str(raw_path).strip():
        return (
            data_root
            / "raw"
            / "tushare"
            / "daily"
            / f"stock={stock_id}.start={start_date.isoformat()}.end={end_date.isoformat()}.json.gz"
        )
    if not isinstance(raw_path, str):
        raise ValueError(f"cache_path at line {line_number} must be a string")
    path = Path(raw_path)
    return path if path.is_absolute() else data_root / path


def _backup_path(cache_path: Path, original_sha256: str) -> Path:
    return cache_path.with_name(f"{cache_path.name}.before-repair.{original_sha256}.bak")


def _write_backup(path: Path, content: bytes, expected_sha256: str) -> None:
    if path.exists():
        if _sha256_file(path) != expected_sha256:
            raise ValueError(f"existing recovery backup has unexpected content: {path}")
        return
    _atomic_write(path, content)


def _gzip_json_bytes(payload: object) -> bytes:
    buffer = BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as stream:
        stream.write(_json_bytes(payload))
    return buffer.getvalue()


def _json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
        stream.write(content)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(content: bytes) -> str:
    return sha256(content).hexdigest()
