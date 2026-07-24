package com.seatflow.backend.graph;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.seatflow.backend.config.GraphProjectionProperties;
import com.seatflow.backend.datacenter.entity.GraphSnapshot;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(prefix = "seatflow.neo4j", name = "enabled", havingValue = "true")
public class Neo4jHttpGraphProjectionPublisher implements GraphProjectionPublisher {
    private static final String SNAPSHOT_AND_NODES = """
            MERGE (snapshot:SeatFlowGraphSnapshot {snapshotId: $snapshotId})
            SET snapshot.provider = $provider,
                snapshot.tradeDate = $tradeDate,
                snapshot.dataScope = $dataScope,
                snapshot.seatDetailScope = $seatDetailScope,
                snapshot.lineageHash = $lineageHash,
                snapshot.nodeCount = $nodeCount,
                snapshot.edgeCount = $edgeCount
            WITH snapshot
            UNWIND $nodes AS node
            MERGE (item:SeatFlowGraphNode {snapshotId: $snapshotId, nodeId: node.nodeId})
            SET item.name = node.name, item.nodeType = node.nodeType, item.value = node.value
            MERGE (snapshot)-[:CONTAINS]->(item)
            """;
    private static final String EDGES = """
            UNWIND $edges AS edge
            MATCH (source:SeatFlowGraphNode {snapshotId: $snapshotId, nodeId: edge.source})
            MATCH (target:SeatFlowGraphNode {snapshotId: $snapshotId, nodeId: edge.target})
            MERGE (source)-[relation:RELATES_TO {snapshotId: $snapshotId, relation: edge.relation}]->(target)
            SET relation.weight = edge.weight
            """;

    private final GraphProjectionProperties properties;
    private final ObjectMapper objectMapper;
    private final HttpClient httpClient;

    public Neo4jHttpGraphProjectionPublisher(GraphProjectionProperties properties, ObjectMapper objectMapper) {
        this.properties = properties;
        this.objectMapper = objectMapper;
        this.httpClient = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();
    }

    @Override
    public void publish(GraphSnapshot snapshot, JsonNode graph) {
        try {
            ObjectNode requestBody = objectMapper.createObjectNode();
            ArrayNode statements = requestBody.putArray("statements");
            statements.add(statement(SNAPSHOT_AND_NODES, snapshotParameters(snapshot, graph)));
            statements.add(statement(EDGES, edgeParameters(snapshot, graph)));
            String authorization = Base64.getEncoder().encodeToString(
                    (properties.username() + ":" + properties.password()).getBytes(StandardCharsets.UTF_8)
            );
            HttpRequest request = HttpRequest.newBuilder(URI.create(properties.baseUrl() + "/db/" + properties.database() + "/tx/commit"))
                    .timeout(Duration.ofSeconds(10))
                    .header("Authorization", "Basic " + authorization)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(requestBody), StandardCharsets.UTF_8))
                    .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            if (response.statusCode() < 200 || response.statusCode() >= 300 || response.body().contains("\"errors\":[{")) {
                throw new IllegalStateException("Neo4j graph projection rejected the snapshot");
            }
        } catch (Exception error) {
            throw new IllegalStateException("Neo4j graph projection failed", error);
        }
    }

    private ObjectNode statement(String cypher, Map<String, Object> parameters) {
        ObjectNode value = objectMapper.createObjectNode();
        value.put("statement", cypher);
        value.set("parameters", objectMapper.valueToTree(parameters));
        return value;
    }

    private Map<String, Object> snapshotParameters(GraphSnapshot snapshot, JsonNode graph) {
        Map<String, Object> values = commonSnapshotParameters(snapshot);
        values.put("nodes", nodes(graph));
        return values;
    }

    private Map<String, Object> edgeParameters(GraphSnapshot snapshot, JsonNode graph) {
        Map<String, Object> values = commonSnapshotParameters(snapshot);
        values.put("edges", edges(graph));
        return values;
    }

    private Map<String, Object> commonSnapshotParameters(GraphSnapshot snapshot) {
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("snapshotId", snapshot.getSnapshotId());
        values.put("provider", snapshot.getProvider());
        values.put("tradeDate", snapshot.getTradeDate().toString());
        values.put("dataScope", snapshot.getDataScope());
        values.put("seatDetailScope", snapshot.getSeatDetailScope());
        values.put("lineageHash", snapshot.getLineageHash());
        values.put("nodeCount", snapshot.getNodeCount());
        values.put("edgeCount", snapshot.getEdgeCount());
        return values;
    }

    private List<Map<String, Object>> nodes(JsonNode graph) {
        List<Map<String, Object>> values = new ArrayList<>();
        if (!graph.path("nodes").isArray()) {
            return values;
        }
        for (JsonNode node : graph.path("nodes")) {
            String nodeId = text(node, "node_id", text(node, "id", null));
            if (nodeId == null) {
                continue;
            }
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("nodeId", nodeId);
            value.put("name", text(node, "name", nodeId));
            value.put("nodeType", text(node, "node_type", text(node, "type", "unknown")));
            value.put("value", node.path("value").asDouble(0));
            values.add(value);
        }
        return values;
    }

    private List<Map<String, Object>> edges(JsonNode graph) {
        List<Map<String, Object>> values = new ArrayList<>();
        if (!graph.path("edges").isArray()) {
            return values;
        }
        for (JsonNode edge : graph.path("edges")) {
            String source = text(edge, "source_id", text(edge, "source", null));
            String target = text(edge, "target_id", text(edge, "target", null));
            if (source == null || target == null) {
                continue;
            }
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("source", source);
            value.put("target", target);
            value.put("relation", text(edge, "relation", "related"));
            value.put("weight", edge.path("weight").asDouble(1));
            values.add(value);
        }
        return values;
    }

    private static String text(JsonNode node, String key, String fallback) {
        JsonNode value = node.path(key);
        return value.isTextual() && !value.asText().isBlank() ? value.asText() : fallback;
    }
}
