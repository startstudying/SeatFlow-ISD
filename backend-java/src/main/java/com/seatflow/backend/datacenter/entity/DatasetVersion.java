package com.seatflow.backend.datacenter.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Version;
import java.time.OffsetDateTime;

@Entity
@Table(name = "sf_dataset_version")
public class DatasetVersion {
    @Id
    private String versionId;
    private String provider;
    private String dataset;
    private String version;
    private String artifactUri;
    private String contentSha256;
    private boolean active;
    @Version
    private long versionLock;
    private OffsetDateTime createdAt;

    protected DatasetVersion() {
    }

    public String getVersionId() { return versionId; }
    public String getProvider() { return provider; }
    public String getDataset() { return dataset; }
    public String getVersion() { return version; }
    public String getArtifactUri() { return artifactUri; }
    public String getContentSha256() { return contentSha256; }
    public boolean isActive() { return active; }
    public long getVersionLock() { return versionLock; }
}
