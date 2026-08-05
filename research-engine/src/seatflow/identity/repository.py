"""Identity repository for persistent storage of stock and seat identities."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from seatflow.identity.stock_identity import StockIdentity, StockIdentityRegistry
from seatflow.identity.seat_identity import SeatIdentity, SeatIdentityRegistry


class IdentityRepository:
    """Repository that wraps stock and seat registries.

    In production this would read from Parquet/PostgreSQL. For now it
    operates in-memory and can be loaded from a manifest.
    """

    def __init__(self) -> None:
        self._stock_registry = StockIdentityRegistry()
        self._seat_registry = SeatIdentityRegistry()

    def register_stock(self, identity: StockIdentity) -> None:
        self._stock_registry.register(identity)

    def register_seat(self, identity: SeatIdentity) -> None:
        self._seat_registry.register(identity)

    def resolve_stock(self, raw_code: str, at_time: datetime) -> StockIdentity:
        return self._stock_registry.resolve_stock(raw_code, at_time)

    def resolve_seat(self, raw_name: str, at_time: datetime):
        return self._seat_registry.resolve_seat(raw_name, at_time)
