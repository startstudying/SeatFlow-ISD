from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import date

from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.graph.temporal import time_decay
from seatflow.graph.types import GraphEdge, NodeType


def build_seat_cooccurrence_edges(
    records: Iterable[DragonTigerRecord],
    *,
    as_of_date: date,
    decay_lambda: float,
    max_edge_age_days: int,
    max_records_per_stock: int | None = None,
) -> tuple[GraphEdge, ...]:
    if max_records_per_stock is not None and max_records_per_stock < 2:
        raise ValueError("max_records_per_stock must be at least two when configured")
    grouped: dict[tuple[date, str], list[DragonTigerRecord]] = defaultdict(list)
    for record in records:
        delta_days = (as_of_date - record.trade_date).days
        if delta_days < 0:
            raise ValueError("records after as_of_date would leak future information")
        if delta_days > max_edge_age_days:
            continue
        grouped[(record.trade_date, record.stock_id)].append(record)

    edges: list[GraphEdge] = []
    for (event_date, _stock_id), bucket in sorted(grouped.items()):
        unique = sorted(bucket, key=lambda item: (item.seat_id, item.side.value, item.amount))
        if max_records_per_stock is not None and len(unique) > max_records_per_stock:
            # A deterministic liquidity proxy keeps pairwise co-occurrence tractable.
            unique = sorted(
                sorted(
                    unique,
                    key=lambda item: (-item.amount, item.seat_id, item.side.value),
                )[:max_records_per_stock],
                key=lambda item: (item.seat_id, item.side.value, item.amount),
            )
        for source in unique:
            for target in unique:
                if source.seat_id == target.seat_id:
                    continue
                delta_days = (as_of_date - event_date).days
                base_amount = min(source.amount, target.amount)
                decay_weight = time_decay(base_amount, delta_days, decay_lambda)
                same_direction = source.side == target.side
                edges.append(
                    GraphEdge(
                        source_type=NodeType.SEAT,
                        source_id=source.seat_id,
                        relation="cooccurs_with",
                        target_type=NodeType.SEAT,
                        target_id=target.seat_id,
                        weight=decay_weight,
                        event_date=event_date,
                        attributes={
                            "amount": base_amount,
                            "source_amount": source.amount,
                            "target_amount": target.amount,
                            "direction": 1.0 if same_direction else -1.0,
                            "source_direction": _side_direction(source.side),
                            "target_direction": _side_direction(target.side),
                            "delta_days": float(delta_days),
                            "time_decay": decay_weight / base_amount if base_amount else 0.0,
                            "seat_profile_available": 0.0,
                            "edge_gate_input": decay_weight,
                        },
                    )
                )
    return tuple(edges)


def _side_direction(side: TradeSide) -> float:
    return 1.0 if side == TradeSide.BUY else -1.0
