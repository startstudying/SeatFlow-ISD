package com.seatflow.backend.settings.repository;

import com.seatflow.backend.settings.entity.UserAccount;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserAccountRepository extends JpaRepository<UserAccount, String> {
}
