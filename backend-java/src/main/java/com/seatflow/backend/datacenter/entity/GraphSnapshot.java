package com.seatflow.backend.datacenter.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Version;
import java.time.LocalDate;

@Entity
@Table(name = "sf_graph_snapshot")
public class GraphSnapshot {
    @Id
    private String snapshotId;
    private String provider;
    private LocalDate tradeDate;
    private long nodeCount;
    private long edgeCount;
    private String dataScope;
    private String seatDetailScope;
    private String lineageHash;
    @Version
    private long version;

    protected GraphSnapshot() {
    }

    public GraphSnapshot(String snapshotId, String provider, LocalDate tradeDate) {
        this.snapshotId = snapshotId;
        this.provider = provider;
        this.tradeDate = tradeDate;
        this.dataScope = "unknown";
        this.seatDetailScope = "unknown";
    }

    public void refresh(long nodeCount, long edgeCount, String dataScope, String seatDetailScope, String lineageHash) {
        this.nodeCount = nodeCount;
        this.edgeCount = edgeCount;
        this.dataScope = dataScope;
        this.seatDetailScope = seatDetailScope;
        this.lineageHash = lineageHash;
    }

    public String getSnapshotId() { return snapshotId; }
    public String getProvider() { return provider; }
    public LocalDate getTradeDate() { return tradeDate; }
    public long getNodeCount() { return nodeCount; }
    public long getEdgeCount() { return edgeCount; }
    public String getDataScope() { return dataScope; }
    public String getSeatDetailScope() { return seatDetailScope; }
    public String getLineageHash() { return lineageHash; }
    public long getVersion() { return version; }
}
