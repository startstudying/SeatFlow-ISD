package com.seatflow.backend.graph;

import com.fasterxml.jackson.databind.JsonNode;
import com.seatflow.backend.datacenter.entity.GraphSnapshot;

public interface GraphProjectionPublisher {
    void publish(GraphSnapshot snapshot, JsonNode graph);
}
