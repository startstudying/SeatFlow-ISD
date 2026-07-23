from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from math import copysign, log1p, sqrt

from seatflow.data.contracts import DailyBar
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.graph.cooccurrence import build_seat_cooccurrence_edges
from seatflow.graph.seat_profile import build_seat_profile_features
from seatflow.graph.stock_correlation import build_stock_correlation_edges
from seatflow.graph.temporal import time_decay
from seatflow.graph.types import GraphEdge, GraphNode, HeteroGraphSnapshot, NodeType
from seatflow.labels import StockLabel


class DynamicGraphBuilder:
    def __init__(
        self,
        decay_lambda: float,
        max_edge_age_days: int,
        *,
        correlation_lookback_days: int = 60,
        correlation_min_overlap: int = 3,
        correlation_min_abs: float = 0.5,
        cooccurrence_max_records_per_stock: int | None = None,
        enable_seat_cooccurrence: bool = True,
        enable_concept_graph: bool = True,
        add_reverse_relations: bool = False,
        aggregate_duplicate_edges: bool = False,
    ) -> None:
        if decay_lambda < 0:
            raise ValueError("decay_lambda must be non-negative")
        if max_edge_age_days <= 0:
            raise ValueError("max_edge_age_days must be greater than zero")
        self.decay_lambda = decay_lambda
        self.max_edge_age_days = max_edge_age_days
        self.correlation_lookback_days = correlation_lookback_days
        self.correlation_min_overlap = correlation_min_overlap
        self.correlation_min_abs = correlation_min_abs
        self.cooccurrence_max_records_per_stock = cooccurrence_max_records_per_stock
        self.enable_seat_cooccurrence = enable_seat_cooccurrence
        self.enable_concept_graph = enable_concept_graph
        self.add_reverse_relations = add_reverse_relations
        self.aggregate_duplicate_edges = aggregate_duplicate_edges

    def build(
        self,
        records: Iterable[DragonTigerRecord],
        as_of_date: date,
        cutoff_time: datetime | None = None,
        *,
        matured_labels: Iterable[StockLabel] = (),
        profile_records: Iterable[DragonTigerRecord] | None = None,
        bars: Iterable[DailyBar] = (),
    ) -> HeteroGraphSnapshot:
        nodes: dict[tuple[NodeType, str], GraphNode] = {}
        edges: list[GraphEdge] = []
        all_records = tuple(records)
        all_bars = tuple(bars)
        seat_profiles = build_seat_profile_features(
            tuple(profile_records) if profile_records is not None else all_records,
            matured_labels,
            as_of_date=as_of_date,
            cutoff_time=cutoff_time,
        )
        stock_features = _stock_features_by_id(
            all_bars,
            as_of_date=as_of_date,
            cutoff_time=cutoff_time,
        )
        eligible_records: list[DragonTigerRecord] = []

        for record in all_records:
            if cutoff_time is not None:
                if record.available_time is None:
                    raise ValueError("record available_time is required when cutoff_time is set")
                if (record.available_time.tzinfo is None) != (cutoff_time.tzinfo is None):
                    raise ValueError(
                        "record and cutoff_time must use compatible timezone awareness"
                    )
                if record.available_time > cutoff_time:
                    continue
            delta_days = (as_of_date - record.trade_date).days
            if delta_days < 0:
                raise ValueError("records after as_of_date would leak future information")
            if delta_days > self.max_edge_age_days:
                continue

            eligible_records.append(record)
            self._add_node(nodes, NodeType.SEAT, record.seat_id, seat_profiles.get(record.seat_id))
            self._add_node(
                nodes,
                NodeType.STOCK,
                record.stock_id,
                stock_features.get(record.stock_id),
            )

            relation = "buys" if record.side == TradeSide.BUY else "sells"
            decay_weight = time_decay(record.amount, delta_days, self.decay_lambda)
            profile_available = 1.0 if record.seat_id in seat_profiles else 0.0
            trade_edge = GraphEdge(
                    source_type=NodeType.SEAT,
                    source_id=record.seat_id,
                    relation=relation,
                    target_type=NodeType.STOCK,
                    target_id=record.stock_id,
                    weight=decay_weight,
                    event_date=record.trade_date,
                    attributes={
                        "amount": record.amount,
                        "direction": 1.0 if record.side == TradeSide.BUY else -1.0,
                        "delta_days": float(delta_days),
                        "time_decay": decay_weight / record.amount,
                        "seat_profile_available": profile_available,
                        "edge_gate_input": decay_weight * (1.0 + profile_available),
                    },
                )
            edges.append(trade_edge)
            if self.add_reverse_relations:
                edges.append(
                    self._reverse_edge(
                        trade_edge,
                        relation="bought_by" if relation == "buys" else "sold_by",
                    )
                )

            if record.industry:
                self._add_static_relation(
                    nodes,
                    edges,
                    record,
                    NodeType.INDUSTRY,
                    record.industry,
                    "belongs_to" if self.add_reverse_relations else "belongs_to_industry",
                    reverse_relation="contains" if self.add_reverse_relations else None,
                )
            if self.enable_concept_graph:
                for concept in record.concepts:
                    self._add_static_relation(
                        nodes,
                        edges,
                        record,
                        NodeType.CONCEPT,
                        concept,
                        "belongs_to_concept",
                    )
            if record.event_reason:
                self._add_static_relation(
                    nodes,
                    edges,
                    record,
                    NodeType.EVENT_REASON,
                    record.event_reason,
                    "triggered_by",
                )

        if self.enable_seat_cooccurrence:
            for edge in build_seat_cooccurrence_edges(
                eligible_records,
                as_of_date=as_of_date,
                decay_lambda=self.decay_lambda,
                max_edge_age_days=self.max_edge_age_days,
                max_records_per_stock=self.cooccurrence_max_records_per_stock,
            ):
                self._add_node(
                    nodes, NodeType.SEAT, edge.source_id, seat_profiles.get(edge.source_id)
                )
                self._add_node(
                    nodes, NodeType.SEAT, edge.target_id, seat_profiles.get(edge.target_id)
                )
                edges.append(self._with_profile_availability(edge, seat_profiles))

        for edge in build_stock_correlation_edges(
            all_bars,
            as_of_date=as_of_date,
            cutoff_time=cutoff_time,
            lookback_days=self.correlation_lookback_days,
            min_overlap=self.correlation_min_overlap,
            min_abs_correlation=self.correlation_min_abs,
        ):
            self._add_node(
                nodes,
                NodeType.STOCK,
                edge.source_id,
                stock_features.get(edge.source_id),
            )
            self._add_node(
                nodes,
                NodeType.STOCK,
                edge.target_id,
                stock_features.get(edge.target_id),
            )
            edges.append(edge)

        if self.aggregate_duplicate_edges:
            edges = self._aggregate_r2_edges(edges)
        ordered_nodes = tuple(
            sorted(nodes.values(), key=lambda node: (node.node_type.value, node.node_id))
        )
        return HeteroGraphSnapshot(as_of_date, ordered_nodes, tuple(edges))

    def _aggregate_r2_edges(self, edges: Sequence[GraphEdge]) -> list[GraphEdge]:
        forward_trade_relations = {"buys", "sells"}
        reverse_trade_relations = {"bought_by", "sold_by"}
        trade_groups: dict[tuple[str, str, str], list[GraphEdge]] = defaultdict(list)
        retained: dict[tuple[object, ...], GraphEdge] = {}
        for edge in edges:
            if edge.relation in reverse_trade_relations:
                continue
            if edge.relation in forward_trade_relations:
                trade_groups[(edge.source_id, edge.target_id, edge.relation)].append(edge)
                continue
            key = (
                edge.source_type,
                edge.source_id,
                edge.relation,
                edge.target_type,
                edge.target_id,
            )
            previous = retained.get(key)
            if previous is None or edge.event_date > previous.event_date:
                retained[key] = edge

        stock_totals: dict[str, float] = defaultdict(float)
        seat_totals: dict[str, float] = defaultdict(float)
        pair_side_totals: dict[tuple[str, str, str], float] = {}
        for key, group in trade_groups.items():
            amount = sum(float(edge.attributes["amount"]) for edge in group)
            pair_side_totals[key] = amount
            stock_totals[key[1]] += amount
            seat_totals[key[0]] += amount

        aggregated = list(retained.values())
        for (seat_id, stock_id, relation), group in sorted(trade_groups.items()):
            amount = pair_side_totals[(seat_id, stock_id, relation)]
            buy_amount = pair_side_totals.get((seat_id, stock_id, "buys"), 0.0)
            sell_amount = pair_side_totals.get((seat_id, stock_id, "sells"), 0.0)
            pair_total = buy_amount + sell_amount
            latest = max(group, key=lambda edge: edge.event_date)
            attributes = dict(latest.attributes)
            attributes.update(
                {
                    "amount": amount,
                    "signed_log_amount_sum": sum(
                        copysign(
                            log1p(abs(float(edge.attributes["amount"]))),
                            float(edge.attributes["amount"]),
                        )
                        for edge in group
                    ),
                    "trade_count": float(len(group)),
                    "last_trade_age": min(
                        float(edge.attributes.get("delta_days", 0.0)) for edge in group
                    ),
                    "amount_share_of_stock": amount / stock_totals[stock_id],
                    "amount_share_of_seat": amount / seat_totals[seat_id],
                    "net_buy_ratio": (
                        (buy_amount - sell_amount) / pair_total if pair_total else 0.0
                    ),
                    "edge_gate_input": sum(edge.weight for edge in group),
                }
            )
            edge = GraphEdge(
                source_type=NodeType.SEAT,
                source_id=seat_id,
                relation=relation,
                target_type=NodeType.STOCK,
                target_id=stock_id,
                weight=sum(item.weight for item in group),
                event_date=latest.event_date,
                attributes=attributes,
            )
            aggregated.append(edge)
            if self.add_reverse_relations:
                aggregated.append(
                    self._reverse_edge(
                        edge,
                        relation="bought_by" if relation == "buys" else "sold_by",
                    )
                )
        return aggregated

    @staticmethod
    def _add_node(
        nodes: dict[tuple[NodeType, str], GraphNode],
        node_type: NodeType,
        node_id: str,
        features: dict[str, float] | None = None,
    ) -> None:
        key = (node_type, node_id)
        if key not in nodes:
            nodes[key] = GraphNode(node_type, node_id, dict(features or {}))
        elif features:
            merged = dict(nodes[key].features)
            merged.update(features)
            nodes[key] = GraphNode(node_type, node_id, merged)

    def _add_static_relation(
        self,
        nodes: dict[tuple[NodeType, str], GraphNode],
        edges: list[GraphEdge],
        record: DragonTigerRecord,
        target_type: NodeType,
        target_id: str,
        relation: str,
        reverse_relation: str | None = None,
    ) -> None:
        self._add_node(nodes, target_type, target_id)
        edge = GraphEdge(
                source_type=NodeType.STOCK,
                source_id=record.stock_id,
                relation=relation,
                target_type=target_type,
                target_id=target_id,
                weight=1.0,
                event_date=record.trade_date,
                attributes={
                    "amount": 0.0,
                    "direction": 0.0,
                    "delta_days": 0.0,
                    "time_decay": 1.0,
                    "seat_profile_available": 0.0,
                    "edge_gate_input": 1.0,
                },
            )
        edges.append(edge)
        if reverse_relation is not None:
            edges.append(self._reverse_edge(edge, relation=reverse_relation))

    @staticmethod
    def _reverse_edge(edge: GraphEdge, *, relation: str) -> GraphEdge:
        return GraphEdge(
            source_type=edge.target_type,
            source_id=edge.target_id,
            relation=relation,
            target_type=edge.source_type,
            target_id=edge.source_id,
            weight=edge.weight,
            event_date=edge.event_date,
            attributes=dict(edge.attributes),
        )

    @staticmethod
    def _with_profile_availability(
        edge: GraphEdge, seat_profiles: dict[str, dict[str, float]]
    ) -> GraphEdge:
        available = (
            1.0 if edge.source_id in seat_profiles or edge.target_id in seat_profiles else 0.0
        )
        attributes = dict(edge.attributes)
        attributes["seat_profile_available"] = available
        attributes["edge_gate_input"] = edge.weight * (1.0 + available)
        return GraphEdge(
            source_type=edge.source_type,
            source_id=edge.source_id,
            relation=edge.relation,
            target_type=edge.target_type,
            target_id=edge.target_id,
            weight=edge.weight,
            event_date=edge.event_date,
            attributes=attributes,
        )


def _stock_features_by_id(
    bars: Iterable[DailyBar],
    *,
    as_of_date: date,
    cutoff_time: datetime | None,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[DailyBar]] = defaultdict(list)
    for bar in bars:
        if bar.trade_date > as_of_date:
            raise ValueError("bars after as_of_date would leak future information")
        if cutoff_time is not None:
            if bar.available_time is None:
                raise ValueError("bar available_time is required when cutoff_time is set")
            if (bar.available_time.tzinfo is None) != (cutoff_time.tzinfo is None):
                raise ValueError("bar and cutoff_time must use compatible timezone awareness")
            if bar.available_time > cutoff_time:
                continue
        grouped[bar.stock_id].append(bar)
    return {
        stock_id: _stock_features(tuple(sorted(stock_bars, key=lambda item: item.trade_date)))
        for stock_id, stock_bars in grouped.items()
        if stock_bars
    }


def _stock_features(ordered_bars: Sequence[DailyBar]) -> dict[str, float]:
    recent = tuple(ordered_bars[-20:])
    current = recent[-1]
    closes = tuple(float(bar.close_price) for bar in recent)
    volumes = tuple(float(bar.volume) for bar in recent)
    highs = tuple(float(bar.high_price) for bar in recent)
    lows = tuple(float(bar.low_price) for bar in recent)
    returns = tuple(
        closes[index] / closes[index - 1] - 1.0
        for index in range(1, len(closes))
        if closes[index - 1] > 0
    )
    mean_volume = sum(volumes) / len(volumes) if volumes else 0.0
    return {
        "stock_bar_count": float(len(recent)),
        "stock_close_to_open": (
            float(current.close_price / current.open_price) - 1.0
            if current.open_price > 0
            else 0.0
        ),
        "stock_return_1d": (
            closes[-1] / closes[-2] - 1.0
            if len(closes) >= 2 and closes[-2] > 0
            else 0.0
        ),
        "stock_return_5d": (
            closes[-1] / closes[-6] - 1.0
            if len(closes) >= 6 and closes[-6] > 0
            else 0.0
        ),
        "stock_return_20d": (
            closes[-1] / closes[0] - 1.0
            if len(closes) >= 20 and closes[0] > 0
            else 0.0
        ),
        "stock_volume_ratio_20d": volumes[-1] / mean_volume if mean_volume > 0 else 0.0,
        "stock_amplitude_20d": (
            (max(highs) - min(lows)) / closes[-1]
            if closes and closes[-1] > 0
            else 0.0
        ),
        "stock_volatility_20d": _std(returns),
    }


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))
