from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

from seatflow.data.contracts import DailyBar
from seatflow.data.providers import MarketDataProvider, ProviderCapabilityError, ProviderError
from seatflow.data.quality import (
    DataQualityReport,
    validate_daily_bars,
    validate_dragon_tiger_records,
    validate_dragon_tiger_summaries,
)
from seatflow.data.records import DragonTigerRecord
from seatflow.data.store import SerializableRecord, SnapshotArtifact, SnapshotStore
from seatflow.features import compute_basic_features
from seatflow.graph.builder import DynamicGraphBuilder
from seatflow.graph.types import HeteroGraphSnapshot


@dataclass(frozen=True)
class IngestionResult:
    provider: str
    trade_date: date
    artifacts: tuple[SnapshotArtifact, ...]
    reports: tuple[DataQualityReport, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "trade_date": self.trade_date.isoformat(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "reports": [report.to_dict() for report in self.reports],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class HistoricalBackfillDayReport:
    trade_date: date
    status: str
    artifacts: tuple[SnapshotArtifact, ...]
    hash_summary: Mapping[str, str]
    reports: tuple[DataQualityReport, ...]
    warnings: tuple[str, ...]
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "trade_date": self.trade_date.isoformat(),
            "status": self.status,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "hash_summary": dict(self.hash_summary),
            "reports": [report.to_dict() for report in self.reports],
            "warnings": list(self.warnings),
        }
        if self.error_type is not None:
            payload["error_type"] = self.error_type
        if self.error_message is not None:
            payload["error_message"] = self.error_message
        return payload


@dataclass(frozen=True)
class HistoricalBackfillResult:
    provider: str
    passed: bool
    trade_dates: tuple[date, ...]
    days: tuple[HistoricalBackfillDayReport, ...]
    continued_on_error: bool
    stopped_early: bool

    def to_dict(self) -> dict[str, object]:
        succeeded = sum(1 for day in self.days if day.status == "ingested")
        skipped = sum(1 for day in self.days if day.status == "skipped")
        failed = sum(1 for day in self.days if day.status == "failed")
        return {
            "provider": self.provider,
            "passed": self.passed,
            "status": "passed" if self.passed else "failed",
            "trade_dates": [item.isoformat() for item in self.trade_dates],
            "day_count": len(self.trade_dates),
            "succeeded": succeeded,
            "skipped": skipped,
            "failed": failed,
            "continued_on_error": self.continued_on_error,
            "stopped_early": self.stopped_early,
            "hash_summary": {
                day.trade_date.isoformat(): dict(day.hash_summary) for day in self.days
            },
            "days": [day.to_dict() for day in self.days],
        }


class DataIngestionService:
    def __init__(self, provider: MarketDataProvider, data_root: str | Path) -> None:
        root = Path(data_root)
        self.root = root
        self.provider = provider
        self.raw_store = SnapshotStore(root / "raw")
        self.processed_store = SnapshotStore(root / "processed")
        self.feature_store = SnapshotStore(root / "features")
        self.graph_store = SnapshotStore(root / "graphs")

    def ingest_market_day(
        self,
        trade_date: date,
        stock_id: str | None = None,
        include_seat_detail: bool = True,
        require_seat_detail: bool = False,
    ) -> IngestionResult:
        if require_seat_detail and not include_seat_detail:
            raise ValueError("require_seat_detail cannot be combined with disabled seat detail")
        artifacts: list[SnapshotArtifact] = []
        reports: list[DataQualityReport] = []
        warnings: list[str] = []

        daily_bars = self.provider.fetch_daily_bars(trade_date, stock_id)
        daily_report = validate_daily_bars(daily_bars)
        reports.append(daily_report)
        self._require_clean(daily_report)
        artifacts.extend(self._write_pair("daily_bar", trade_date, daily_bars, daily_report))

        summaries = self.provider.fetch_dragon_tiger_summaries(trade_date)
        summary_report = validate_dragon_tiger_summaries(summaries)
        reports.append(summary_report)
        self._require_clean(summary_report)
        artifacts.extend(
            self._write_pair("dragon_tiger_summary", trade_date, summaries, summary_report)
        )

        if include_seat_detail:
            try:
                seat_events = self.provider.fetch_dragon_tiger_seat_events(trade_date)
            except ProviderCapabilityError as error:
                if require_seat_detail:
                    raise
                warnings.append(str(error))
            else:
                event_report = validate_dragon_tiger_records(seat_events)
                reports.append(event_report)
                self._require_clean(event_report)
                artifacts.extend(
                    self._write_pair(
                        "dragon_tiger_seat_event", trade_date, seat_events, event_report
                    )
                )
                if not seat_events:
                    message = "provider returned no seat-detail events for this trading day"
                    if require_seat_detail:
                        raise ProviderCapabilityError(message)
                    warnings.append(message)

        return IngestionResult(
            provider=self.provider.name,
            trade_date=trade_date,
            artifacts=tuple(artifacts),
            reports=tuple(reports),
            warnings=tuple(warnings),
        )

    def backfill_history(
        self,
        trade_dates: Iterable[date],
        *,
        stock_id: str | None = None,
        include_seat_detail: bool = True,
        require_seat_detail: bool = False,
        skip_verified: bool = True,
        continue_on_error: bool = False,
    ) -> HistoricalBackfillResult:
        if require_seat_detail and not include_seat_detail:
            raise ValueError("require_seat_detail cannot be combined with disabled seat detail")
        ordered_dates = tuple(trade_dates)
        if not ordered_dates:
            raise ValueError("historical backfill requires at least one trade date")
        if len(set(ordered_dates)) != len(ordered_dates):
            raise ValueError("historical backfill trade dates must be unique")

        days: list[HistoricalBackfillDayReport] = []
        stopped_early = False
        for trade_date in ordered_dates:
            if skip_verified:
                verified_hashes = self._verified_snapshot_hashes(
                    trade_date,
                    include_seat_detail=include_seat_detail,
                )
                if verified_hashes is not None:
                    days.append(
                        HistoricalBackfillDayReport(
                            trade_date=trade_date,
                            status="skipped",
                            artifacts=(),
                            hash_summary=verified_hashes,
                            reports=(),
                            warnings=("verified snapshots already exist; ingestion skipped",),
                        )
                    )
                    continue
            try:
                result = self.ingest_market_day(
                    trade_date,
                    stock_id=stock_id,
                    include_seat_detail=include_seat_detail,
                    require_seat_detail=require_seat_detail,
                )
            except (ProviderError, ValueError) as error:
                days.append(
                    HistoricalBackfillDayReport(
                        trade_date=trade_date,
                        status="failed",
                        artifacts=(),
                        hash_summary={},
                        reports=(),
                        warnings=(),
                        error_type=type(error).__name__,
                        error_message=_safe_error_message(error),
                    )
                )
                if not continue_on_error:
                    stopped_early = True
                    break
            else:
                days.append(
                    HistoricalBackfillDayReport(
                        trade_date=trade_date,
                        status="ingested",
                        artifacts=result.artifacts,
                        hash_summary=_artifact_hash_summary(result.artifacts),
                        reports=result.reports,
                        warnings=result.warnings,
                    )
                )

        passed = len(days) == len(ordered_dates) and all(day.status != "failed" for day in days)
        return HistoricalBackfillResult(
            provider=self.provider.name,
            passed=passed,
            trade_dates=ordered_dates,
            days=tuple(days),
            continued_on_error=continue_on_error,
            stopped_early=stopped_early,
        )

    def build_features(self) -> tuple[SnapshotArtifact, ...]:
        raw_rows = self.processed_store.read_all(self.provider.name, "daily_bar")
        bars = tuple(DailyBar.from_dict(dict(row)) for row in raw_rows)
        if not bars:
            return ()
        features = compute_basic_features(bars)
        artifacts: list[SnapshotArtifact] = []
        for trade_date in sorted({feature.trade_date for feature in features}):
            features_for_date = tuple(
                feature for feature in features if feature.trade_date == trade_date
            )
            artifacts.append(
                self.feature_store.write(
                    self.provider.name,
                    "stock_feature",
                    trade_date,
                    features_for_date,
                    metadata={"source_dataset": "daily_bar", "feature_version": "basic-v1"},
                )
            )
        return tuple(artifacts)

    def build_graph_snapshot(
        self,
        as_of_date: date,
        decay_lambda: float,
        max_edge_age_days: int,
        cutoff_time: datetime | None = None,
    ) -> SnapshotArtifact | None:
        raw_rows = self.processed_store.read_all(self.provider.name, "dragon_tiger_seat_event")
        records = tuple(_seat_record_from_dict(dict(row)) for row in raw_rows)
        records = tuple(record for record in records if record.trade_date <= as_of_date)
        if cutoff_time is not None:
            records = tuple(
                record
                for record in records
                if record.available_time is None or record.available_time <= cutoff_time
            )
        if not records:
            return None
        graph = DynamicGraphBuilder(decay_lambda, max_edge_age_days).build(
            records, as_of_date, cutoff_time=cutoff_time
        )
        return self.graph_store.write(
            self.provider.name,
            "hetero_graph",
            as_of_date,
            [_graph_to_dict(graph)],
            metadata={
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "decay_lambda": decay_lambda,
                "max_edge_age_days": max_edge_age_days,
                "data_scope": _default_graph_data_scope(self.provider),
                "seat_detail_scope": self.provider.status().seat_detail_scope,
                "risk_flags": ["data_audit_not_verified"],
            },
        )

    def promote_graph_snapshot(
        self,
        *,
        as_of_date: date,
        audit_report_path: str | Path,
        require_complete_seat: bool = False,
    ) -> SnapshotArtifact:
        """Attach verified audit evidence without changing the persisted graph data."""

        provider_status = self.provider.status()
        if provider_status.seat_detail_scope == "synthetic":
            raise ValueError("synthetic providers cannot promote a graph to real data")
        audit_path = Path(audit_report_path)
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("audit report is unreadable") from error
        if not isinstance(audit, dict):
            raise ValueError("audit report must be a JSON object")
        provider = audit.get("provider_status")
        if not isinstance(provider, dict) or provider.get("provider") != self.provider.name:
            raise ValueError("audit report provider does not match graph provider")
        scope = audit.get("seat_detail_scope")
        if scope not in {"complete_seat", "institutional_only"}:
            raise ValueError("audit report does not contain a promotable seat_detail_scope")
        if audit.get("data_readiness") is not True or audit.get("data_scope") != "real":
            raise ValueError("audit report does not prove real-data readiness")
        if require_complete_seat and scope != "complete_seat":
            raise ValueError("complete_seat evidence is required for this graph promotion")

        output_dir = self.root / "graphs" / self.provider.name / "hetero_graph"
        output_dir = output_dir / f"date={as_of_date.isoformat()}"
        data_path = output_dir / "data.json"
        manifest_path = output_dir / "manifest.json"
        if not data_path.exists() or not manifest_path.exists():
            raise ValueError("graph snapshot is missing and cannot be promoted")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("graph manifest must be a JSON object")
        data_bytes = data_path.read_bytes()
        data_hash = sha256(data_bytes).hexdigest()
        if manifest.get("sha256") != data_hash:
            raise ValueError("graph data hash does not match its manifest")
        metadata = manifest.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.update(
            {
                "data_scope": "real",
                "seat_detail_scope": scope,
                "data_audit_report_sha256": sha256(audit_path.read_bytes()).hexdigest(),
                "risk_flags": [] if scope == "complete_seat" else ["institutional_only_scope"],
            }
        )
        manifest["metadata"] = metadata
        manifest["promoted_at"] = datetime.now().astimezone().isoformat()
        _atomic_json_write(manifest_path, manifest)
        return SnapshotArtifact(
            data_path=data_path,
            manifest_path=manifest_path,
            row_count=int(manifest.get("row_count", 0)),
            sha256=data_hash,
        )

    def _write_pair(
        self,
        dataset: str,
        trade_date: date,
        rows: tuple[SerializableRecord, ...],
        report: DataQualityReport,
    ) -> tuple[SnapshotArtifact, SnapshotArtifact]:
        metadata: Mapping[str, object] = {"quality": report.to_dict()}
        raw = self.raw_store.write(self.provider.name, dataset, trade_date, rows, metadata)
        processed = self.processed_store.write(
            self.provider.name, dataset, trade_date, rows, metadata
        )
        return raw, processed

    @staticmethod
    def _require_clean(report: DataQualityReport) -> None:
        if not report.passed:
            raise ValueError(
                f"data quality failed for {report.dataset}: {'; '.join(report.errors)}"
            )

    def _verified_snapshot_hashes(
        self,
        trade_date: date,
        *,
        include_seat_detail: bool,
    ) -> dict[str, str] | None:
        datasets = ["daily_bar", "dragon_tiger_summary"]
        if include_seat_detail:
            datasets.append("dragon_tiger_seat_event")
        hashes: dict[str, str] = {}
        for store_name, store in (("raw", self.raw_store), ("processed", self.processed_store)):
            for dataset in datasets:
                manifest_path = (
                    store.root
                    / self.provider.name
                    / dataset
                    / f"date={trade_date.isoformat()}"
                    / "manifest.json"
                )
                data_path = manifest_path.with_name("data.json")
                digest = _verified_snapshot_hash(
                    manifest_path,
                    data_path,
                    provider=self.provider.name,
                    dataset=dataset,
                    trade_date=trade_date,
                )
                if digest is None:
                    return None
                hashes[f"{store_name}/{dataset}"] = digest
        return hashes


def _seat_record_from_dict(raw: dict[str, object]) -> DragonTigerRecord:
    from datetime import datetime

    from seatflow.data.contracts import parse_trade_date
    from seatflow.data.records import TradeSide

    available_time = raw.get("available_time")
    return DragonTigerRecord(
        trade_date=parse_trade_date(str(raw["trade_date"])),
        seat_id=str(raw["seat_id"]),
        stock_id=str(raw["stock_id"]),
        side=TradeSide(str(raw["side"])),
        amount=float(str(raw["amount"])),
        industry=_optional_string(raw.get("industry")),
        concepts=_string_tuple(raw.get("concepts")),
        event_reason=_optional_string(raw.get("event_reason")),
        available_time=datetime.fromisoformat(str(available_time)) if available_time else None,
        source_name=str(raw.get("source_name") or "unknown"),
        source_version=_optional_string(raw.get("source_version")),
        source_row_count=int(str(raw.get("source_row_count", 1))),
        source_side_set=_string_tuple(raw.get("source_side_set")),
        identity_scope=str(raw.get("identity_scope") or "named_seat"),
    )


def _optional_string(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _string_tuple(value: object | None) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _graph_to_dict(graph: HeteroGraphSnapshot) -> dict[str, object]:
    return {
        "as_of_date": graph.as_of_date.isoformat(),
        "nodes": [
            {
                "node_type": node.node_type.value,
                "node_id": node.node_id,
                "features": node.features,
            }
            for node in graph.nodes
        ],
        "edges": [
            {
                "source_type": edge.source_type.value,
                "source_id": edge.source_id,
                "relation": edge.relation,
                "target_type": edge.target_type.value,
                "target_id": edge.target_id,
                "weight": edge.weight,
                "event_date": edge.event_date.isoformat(),
                "attributes": edge.attributes,
            }
            for edge in graph.edges
        ],
    }


def write_historical_backfill_report(
    path: str | Path, result: HistoricalBackfillResult
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(output, result.to_dict())
    return output


def write_historical_backfill_summary(
    path: str | Path, result: HistoricalBackfillResult
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(
        output,
        {
            "provider": result.provider,
            "passed": result.passed,
            "status": "passed" if result.passed else "failed",
            "day_count": len(result.trade_dates),
            "succeeded": sum(1 for day in result.days if day.status == "ingested"),
            "skipped": sum(1 for day in result.days if day.status == "skipped"),
            "failed": sum(1 for day in result.days if day.status == "failed"),
            "stopped_early": result.stopped_early,
            "days": [
                {
                    "trade_date": day.trade_date.isoformat(),
                    "status": day.status,
                    "error_type": day.error_type,
                    "error_message": day.error_message,
                    "hash_summary": dict(day.hash_summary),
                }
                for day in result.days
            ],
        },
    )
    return output


def _artifact_hash_summary(artifacts: Iterable[SnapshotArtifact]) -> dict[str, str]:
    summary: dict[str, str] = {}
    for artifact in artifacts:
        manifest_path = artifact.manifest_path
        parts = manifest_path.parts
        if len(parts) >= 4:
            key = f"{parts[-5]}/{parts[-3]}" if len(parts) >= 5 else str(manifest_path)
        else:
            key = str(manifest_path)
        summary[key] = artifact.sha256
    return summary


def _verified_snapshot_hash(
    manifest_path: Path,
    data_path: Path,
    *,
    provider: str,
    dataset: str,
    trade_date: date,
) -> str | None:
    if not manifest_path.exists() or not data_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data = json.loads(data_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or not isinstance(data, list):
        return None
    data_hash = sha256(data_path.read_bytes()).hexdigest()
    metadata = manifest.get("metadata")
    quality = metadata.get("quality") if isinstance(metadata, dict) else None
    if (
        manifest.get("provider") != provider
        or manifest.get("dataset") != dataset
        or manifest.get("trade_date") != trade_date.isoformat()
        or manifest.get("sha256") != data_hash
        or manifest.get("row_count") != len(data)
        or not isinstance(quality, dict)
        or quality.get("passed") is not True
    ):
        return None
    return data_hash


def _safe_error_message(error: BaseException) -> str:
    message = str(error)
    message = re.sub(r"(?i)(token|api[_-]?key|secret|password)=\S+", r"\1=<redacted>", message)
    message = re.sub(r"(?i)(token|api[_-]?key|secret|password):\s*\S+", r"\1: <redacted>", message)
    return message


def _default_graph_data_scope(provider: MarketDataProvider) -> str:
    return "synthetic" if provider.status().seat_detail_scope == "synthetic" else "none"


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    with NamedTemporaryFile("wb", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
        stream.write(content)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)
