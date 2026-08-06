"""Data provider interface and abstract base."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from seatflow.ingestion.models import BronzeRecord


class ProviderCapability:
    """Capability status for a data source adapter."""

    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


class CapabilityReason:
    """Stable reason codes for machine-readable provider degradation."""

    COMPLETE = "COMPLETE"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    MISSING_REQUIRED_FIELDS = "MISSING_REQUIRED_FIELDS"
    AMBIGUOUS_FIELD_ALIASES = "AMBIGUOUS_FIELD_ALIASES"
    PROXY_COVERAGE_ONLY = "PROXY_COVERAGE_ONLY"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"


@dataclass(frozen=True)
class ProviderCapabilityEvidence:
    """Evidence explaining a provider dataset's current capability."""

    status: str
    reason_code: str
    required_fields: tuple[str, ...] = ()
    present_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    field_coverage: float = 0.0
    proxy_only: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)


class ProviderFieldError(ValueError):
    """Raised when a provider row cannot satisfy its explicit field contract."""

    def __init__(
        self,
        *,
        dataset: str,
        reason_code: str,
        missing_fields: tuple[str, ...] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.reason_code = reason_code
        self.missing_fields = missing_fields
        self.details = dict(details or {})
        super().__init__(
            f"{reason_code}: provider field contract failed for {dataset}; "
            f"missing={list(missing_fields)}"
        )


def canonicalize_provider_fields(
    row: Mapping[str, Any],
    *,
    canonical_fields: Mapping[str, str],
    built_in_aliases: Mapping[str, tuple[str, ...]],
    configured_aliases: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """Copy a provider row while resolving only declared UTF-8 aliases."""
    payload = dict(row)
    present: list[str] = []
    missing: list[str] = []
    for semantic_name, canonical_name in canonical_fields.items():
        aliases = (
            canonical_name,
            *built_in_aliases.get(semantic_name, ()),
            *configured_aliases.get(semantic_name, ()),
        )
        found = [
            (alias, payload[alias])
            for alias in dict.fromkeys(aliases)
            if payload.get(alias) not in (None, "")
        ]
        values = {repr(value) for _, value in found}
        if len(values) > 1:
            raise ProviderFieldError(
                dataset="unknown",
                reason_code=CapabilityReason.AMBIGUOUS_FIELD_ALIASES,
                details={
                    "semantic_field": semantic_name,
                    "aliases": [name for name, _ in found],
                },
            )
        if not found:
            missing.append(semantic_name)
            continue
        payload[canonical_name] = found[0][1]
        present.append(semantic_name)
    return payload, tuple(present), tuple(missing)


class DataProvider(ABC):
    """Abstract data provider that yields Bronze records.

    Adapters must implement each ``fetch_*`` method. They may return
    ``ProviderCapability.DEGRADED`` when the underlying source is partially
    available. No adapter may read environment variables at import time.
    """

    @abstractmethod
    def capability(self) -> dict[str, str]:
        """Return a map of dataset -> capability status."""

    @abstractmethod
    def capability_evidence(self) -> dict[str, ProviderCapabilityEvidence]:
        """Return structured status, coverage, and limitation evidence."""

    @abstractmethod
    def fetch_daily_bars(self, *args: Any, **kwargs: Any) -> Iterable[BronzeRecord]:
        """Fetch daily OHLCV bars."""

    @abstractmethod
    def fetch_stock_master(self, *args: Any, **kwargs: Any) -> Iterable[BronzeRecord]:
        """Fetch stock master (listings, delistings)."""

    @abstractmethod
    def fetch_trade_calendar(self, *args: Any, **kwargs: Any) -> Iterable[BronzeRecord]:
        """Fetch trade calendar."""

    @abstractmethod
    def fetch_lhb_summary(self, *args: Any, **kwargs: Any) -> Iterable[BronzeRecord]:
        """Fetch Dragon-Tiger-List event summaries."""

    @abstractmethod
    def fetch_lhb_participation(self, *args: Any, **kwargs: Any) -> Iterable[BronzeRecord]:
        """Fetch Dragon-Tiger-List seat participation details."""

    @abstractmethod
    def fetch_memberships(self, *args: Any, **kwargs: Any) -> Iterable[BronzeRecord]:
        """Fetch industry/concept memberships."""

    @abstractmethod
    def fetch_tradeability(self, *args: Any, **kwargs: Any) -> Iterable[BronzeRecord]:
        """Fetch tradeability states (suspension, limit-up/down, delisting)."""
