package com.seatflow.backend.graph;

import com.fasterxml.jackson.databind.JsonNode;
import com.seatflow.backend.datacenter.entity.GraphSnapshot;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(prefix = "seatflow.neo4j", name = "enabled", havingValue = "false", matchIfMissing = true)
public class NoopGraphProjectionPublisher implements GraphProjectionPublisher {
    @Override
    public void publish(GraphSnapshot snapshot, JsonNode graph) {
        // MySQL retains snapshot metadata while the optional projection is disabled.
    }
}
