"""Tests for graph schema (Task 14)."""

from seatflow.graph.schema import (
    EdgeType,
    NodeType,
    NODE_SCHEMAS,
    EDGE_SCHEMAS,
    get_node_schema,
    get_edge_schema,
    max_propagation_hops,
    default_top_k,
)


class TestNodeSchema:
    def test_all_node_types_have_schemas(self):
        for nt in NodeType:
            schema = get_node_schema(nt)
            assert schema.node_type == nt

    def test_stock_node_has_stock_id(self):
        schema = get_node_schema(NodeType.STOCK)
        assert "stock_id" in schema.properties
        assert "stock_id" in schema.required

    def test_seat_node_has_seat_type(self):
        schema = get_node_schema(NodeType.SEAT)
        assert "seat_type" in schema.properties

    def test_event_node_has_event_id(self):
        schema = get_node_schema(NodeType.EVENT)
        assert "event_id" in schema.properties
        assert "event_id" in schema.required


class TestEdgeSchema:
    def test_all_edge_types_have_schemas(self):
        for et in EdgeType:
            schema = get_edge_schema(et)
            assert schema.edge_type == et

    def test_propagation_edges_max_2_hops(self):
        for et in [EdgeType.SAME_SEAT_RELAY, EdgeType.SAME_COALITION_RELAY,
                    EdgeType.SAME_THEME_DIFFUSION, EdgeType.CAPITAL_ROTATION,
                    EdgeType.SPILLOVER_CANDIDATE]:
            schema = get_edge_schema(et)
            assert schema.max_hops <= 2

    def test_propagation_edges_have_top_k(self):
        for et in [EdgeType.SAME_SEAT_RELAY, EdgeType.SAME_COALITION_RELAY,
                    EdgeType.SAME_THEME_DIFFUSION, EdgeType.CAPITAL_ROTATION,
                    EdgeType.SPILLOVER_CANDIDATE]:
            schema = get_edge_schema(et)
            assert schema.top_k > 0
            assert schema.top_k <= 100

    def test_theme_edge_max_1_hop(self):
        schema = get_edge_schema(EdgeType.STOCK_BELONGS_THEME)
        assert schema.max_hops == 1

    def test_seat_buy_event_connects_seat_to_event(self):
        schema = get_edge_schema(EdgeType.SEAT_BUY_EVENT)
        assert schema.source_type == NodeType.SEAT
        assert schema.target_type == NodeType.EVENT


class TestGlobalConstraints:
    def test_max_propagation_hops_is_2(self):
        assert max_propagation_hops() == 2

    def test_default_top_k(self):
        assert default_top_k() == 20
