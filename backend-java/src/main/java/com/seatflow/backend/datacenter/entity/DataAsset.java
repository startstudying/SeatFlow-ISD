package com.seatflow.backend.datacenter.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Version;
import java.time.OffsetDateTime;

@Entity
@Table(name = "sf_data_asset")
public class DataAsset {
    @Id
    private String assetId;
    private String provider;
    private String dataset;
    private String store;
    private String dataScope;
    private String seatDetailScope;
    private String lineageHash;
    private long rowCount;
    @Version
    private long version;
    private OffsetDateTime createdAt;
    private OffsetDateTime updatedAt;

    protected DataAsset() {
    }

    public DataAsset(String assetId, String provider, String dataset, String store, long rowCount) {
        this.assetId = assetId;
        this.provider = provider;
        this.dataset = dataset;
        this.store = store;
        this.rowCount = rowCount;
        this.dataScope = "unknown";
        this.seatDetailScope = "unknown";
    }

    public void refresh(long rowCount, String dataScope, String seatDetailScope, String lineageHash) {
        this.rowCount = rowCount;
        this.dataScope = dataScope;
        this.seatDetailScope = seatDetailScope;
        this.lineageHash = lineageHash;
    }

    public String getAssetId() { return assetId; }
    public String getProvider() { return provider; }
    public String getDataset() { return dataset; }
    public String getStore() { return store; }
    public String getDataScope() { return dataScope; }
    public String getSeatDetailScope() { return seatDetailScope; }
    public String getLineageHash() { return lineageHash; }
    public long getRowCount() { return rowCount; }
    public long getVersion() { return version; }
}
