package com.seatflow.backend.datacenter.repository;

import com.seatflow.backend.datacenter.entity.DataAsset;
import java.util.Optional;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DataAssetRepository extends JpaRepository<DataAsset, String> {
    Page<DataAsset> findByDatasetContainingIgnoreCaseOrProviderContainingIgnoreCase(
            String dataset, String provider, Pageable pageable
    );

    Optional<DataAsset> findByProviderAndDatasetAndStore(String provider, String dataset, String store);
}
