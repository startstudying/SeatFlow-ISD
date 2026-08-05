"""Data provider interface and abstract base."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from seatflow.ingestion.models import BronzeRecord


class ProviderCapability:
    """Capability status for a data source adapter."""

    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


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
    def fetch_daily_bars(self, *args, **kwargs) -> Iterable[BronzeRecord]:
        """Fetch daily OHLCV bars."""

    @abstractmethod
    def fetch_stock_master(self, *args, **kwargs) -> Iterable[BronzeRecord]:
        """Fetch stock master (listings, delistings)."""

    @abstractmethod
    def fetch_trade_calendar(self, *args, **kwargs) -> Iterable[BronzeRecord]:
        """Fetch trade calendar."""

    @abstractmethod
    def fetch_lhb_summary(self, *args, **kwargs) -> Iterable[BronzeRecord]:
        """Fetch Dragon-Tiger-List event summaries."""

    @abstractmethod
    def fetch_lhb_participation(self, *args, **kwargs) -> Iterable[BronzeRecord]:
        """Fetch Dragon-Tiger-List seat participation details."""

    @abstractmethod
    def fetch_memberships(self, *args, **kwargs) -> Iterable[BronzeRecord]:
        """Fetch industry/concept memberships."""

    @abstractmethod
    def fetch_tradeability(self, *args, **kwargs) -> Iterable[BronzeRecord]:
        """Fetch tradeability states (suspension, limit-up/down, delisting)."""
