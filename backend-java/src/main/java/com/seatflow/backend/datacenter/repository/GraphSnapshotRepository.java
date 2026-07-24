package com.seatflow.backend.datacenter.repository;

import com.seatflow.backend.datacenter.entity.GraphSnapshot;
import java.time.LocalDate;
import java.util.Optional;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface GraphSnapshotRepository extends JpaRepository<GraphSnapshot, String> {
    Optional<GraphSnapshot> findByProviderAndTradeDate(String provider, LocalDate tradeDate);
    Page<GraphSnapshot> findByProviderContainingIgnoreCase(String provider, Pageable pageable);
}
