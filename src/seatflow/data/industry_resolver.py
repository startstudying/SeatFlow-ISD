"""Week-2 fix: resolve stock_id -> SW2021 L1 industry for seat-event records.

The Phase-2 industry membership is backfilled by
``scripts/backfill_r1_industry_membership.py`` into point-in-time snapshots,
but ``DragonTigerRecord.industry`` is left ``None`` because nothing joins the
two on the seat-event construction path. This is why E1 equals E0: the v3
model's industry branch is wired (``use_industry_relations=True``) but the
industry node set is empty, so the branch contributes nothing.

This module provides a pure resolver that loads the SW2021 membership
snapshot and answers ``industry_for(stock_id, as_of)`` in a point-in-time
way (a stock is a member only if ``in_date <= as_of < out_date``). The
caller — the seat-event builder — uses it to populate
``DragonTigerRecord.industry`` so the graph builder sees non-empty industry
nodes.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class IndustryMembership:
    """A single point-in-time SW2021 L1 membership row."""

    stock_id: str
    industry_code: str
    industry_name: str
    in_date: date
    out_date: date | None

    def is_active(self, as_of: date) -> bool:
        if as_of < self.in_date:
            return False
        if self.out_date is not None and as_of >= self.out_date:
            return False
        return True


@dataclass(frozen=True)
class IndustryResolver:
    """Point-in-time SW2021 L1 industry resolver.

    Built once from a membership table (CSV or rows) and queried per
    stock-day. ``industry_for`` returns the L1 industry name for a stock on
    a given date, or ``None`` if the stock has no active membership. When a
    stock has multiple active memberships (overlapping snapshots), the one
    with the latest ``in_date`` wins.
    """

    memberships: tuple[IndustryMembership, ...]

    def industry_for(self, stock_id: str, as_of: date) -> str | None:
        best: IndustryMembership | None = None
        for membership in self._memberships_for(stock_id):
            if not membership.is_active(as_of):
                continue
            if best is None or membership.in_date > best.in_date:
                best = membership
        return best.industry_name if best is not None else None

    def resolve_records(
        self,
        records: Iterable[tuple[str, date]],
    ) -> Mapping[str, Mapping[str, str | None]]:
        """Bulk-resolve (stock_id, as_of) pairs to industry names.

        Returns a nested dict ``{stock_id: {as_of_iso: industry_or_None}}``
        so a seat-event builder can fill many records without re-scanning the
        full membership table per record.
        """
        resolved: dict[str, dict[str, str | None]] = {}
        for stock_id, as_of in records:
            by_date = resolved.setdefault(stock_id, {})
            by_date[as_of.isoformat()] = self.industry_for(stock_id, as_of)
        return resolved

    def _memberships_for(self, stock_id: str) -> tuple[IndustryMembership, ...]:
        # Linear scan is fine for the research-scale membership table
        # (~10k rows). If this becomes hot, index by stock_id at build time.
        return tuple(
            membership
            for membership in self.memberships
            if membership.stock_id == stock_id
        )


def load_industry_resolver(csv_path: str | Path) -> IndustryResolver:
    """Build an IndustryResolver from a SW2021 membership CSV.

    Expected columns: ``ts_code,l1_code,l1_name,in_date,out_date`` (the
    schema written by ``backfill_r1_industry_membership.py``). ``out_date``
    may be empty for active memberships.
    """
    memberships: list[IndustryMembership] = []
    with Path(csv_path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            stock_id = (row.get("ts_code") or "").strip()
            l1_code = (row.get("l1_code") or "").strip()
            l1_name = (row.get("l1_name") or "").strip()
            in_date_raw = (row.get("in_date") or "").strip()
            out_date_raw = (row.get("out_date") or "").strip()
            if not stock_id or not l1_name or not in_date_raw:
                continue
            memberships.append(
                IndustryMembership(
                    stock_id=stock_id,
                    industry_code=l1_code,
                    industry_name=l1_name,
                    in_date=date.fromisoformat(in_date_raw),
                    out_date=(
                        date.fromisoformat(out_date_raw)
                        if out_date_raw and out_date_raw.lower() != "nan"
                        else None
                    ),
                )
            )
    return IndustryResolver(memberships=tuple(memberships))
