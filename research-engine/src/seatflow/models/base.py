"""Formal Base daily-path model and input contract.

Base predicts one five-trading-day daily return-increment path. The
1/3/5 cumulative horizons are always derived from that path, never from
independent heads.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence, cast

import torch
import torch.nn as nn

from seatflow.features.registry import FeatureScope, FeatureSpec, SourceKind
from seatflow.features.registry import is_forbidden_identity_feature_name


BASE_DAILY_PATH_DAYS = 5
BASE_CUMULATIVE_HORIZONS = (1, 3, 5)
BASE_FORMAL_CLASSIFICATION = "FORMAL_BASE_DAILY_PATH"
BASE_PROVENANCE = "BASE_ONLY"
_BASE_ALLOWED_SCOPES = frozenset({FeatureScope.STOCK_PIT, FeatureScope.MARKET_PIT})
_BASE_ALLOWED_SOURCES = frozenset({SourceKind.STOCK_PIT, SourceKind.MARKET_PIT})
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ISSUANCE_SENTINEL = object()


def _validate_daily_path_tensor(daily_path: torch.Tensor) -> None:
    if daily_path.ndim != 2 or daily_path.shape[1] != BASE_DAILY_PATH_DAYS:
        raise ValueError("Base daily path must have shape [batch, 5]")
    if not torch.isfinite(daily_path).all():
        raise ValueError("Base daily path must contain only finite values")


def cumulative_from_daily_path(daily_path: torch.Tensor) -> torch.Tensor:
    """Derive cumulative t+1/t+3/t+5 from one five-day daily path."""
    _validate_daily_path_tensor(daily_path)
    return torch.stack(
        [
            daily_path[:, 0],
            daily_path[:, :3].sum(dim=1),
            daily_path[:, :5].sum(dim=1),
        ],
        dim=1,
    )


def _cumulative_tuple(daily_path: Sequence[float]) -> tuple[float, float, float]:
    return (
        daily_path[0],
        sum(daily_path[:3]),
        sum(daily_path[:5]),
    )


@dataclass(frozen=True)
class BasePrediction:
    """Immutable Base prediction DTO.

    ``stock_id`` is metadata only. Numeric predictions are the five-day
    daily path and the derived cumulative 1/3/5 values.
    """

    stock_id: str
    prediction_time: datetime
    cutoff_time: datetime
    daily_path: tuple[float, float, float, float, float]
    feature_schema_fingerprint: str
    cumulative: tuple[float, float, float] | None = None
    classification: str = field(default=BASE_FORMAL_CLASSIFICATION, init=False)
    provenance: str = field(default=BASE_PROVENANCE, init=False)

    def __post_init__(self) -> None:
        if not self.stock_id:
            raise ValueError("BasePrediction.stock_id metadata must be non-empty")
        _validate_aware_datetime(self.prediction_time, "prediction_time")
        _validate_aware_datetime(self.cutoff_time, "cutoff_time")
        if self.prediction_time <= self.cutoff_time:
            raise ValueError("BasePrediction prediction_time must be after cutoff_time")
        if not _SHA256_RE.fullmatch(self.feature_schema_fingerprint):
            raise ValueError("BasePrediction feature_schema_fingerprint must be sha256:<64 hex>")
        if len(self.daily_path) != BASE_DAILY_PATH_DAYS:
            raise ValueError("BasePrediction daily path must contain five values")
        if not all(math.isfinite(v) for v in self.daily_path):
            raise ValueError("BasePrediction daily path must contain only finite values")

        derived = _cumulative_tuple(self.daily_path)
        if self.cumulative is None:
            object.__setattr__(self, "cumulative", derived)
        elif len(self.cumulative) != len(derived) or any(
            not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
            for a, b in zip(self.cumulative, derived)
        ):
            raise ValueError("BasePrediction cumulative values must match the daily path")


def _validate_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"BasePrediction {field_name} must be timezone-aware")


def _tensor_content_fingerprint(tensor: torch.Tensor) -> str:
    detached = tensor.detach().contiguous().cpu()
    payload = detached.numpy().tobytes()
    header = json.dumps(
        {
            "dtype": str(detached.dtype),
            "shape": list(detached.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header + b"\0" + payload).hexdigest()
    return "sha256:" + digest


@dataclass(frozen=True, init=False)
class BaseFeatureBatch:
    """Frozen Base input batch already validated against a feature contract."""

    features: torch.Tensor
    stock_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    schema_fingerprint: str
    feature_content_fingerprint: str
    _issued_by_contract: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("BaseFeatureBatch must be issued by BaseFeatureContract.validate_batch")

    @classmethod
    def _issue(
        cls,
        *,
        issuance_sentinel: object,
        features: torch.Tensor,
        stock_ids: tuple[str, ...],
        feature_names: tuple[str, ...],
        schema_fingerprint: str,
    ) -> BaseFeatureBatch:
        if issuance_sentinel is not _ISSUANCE_SENTINEL:
            raise TypeError("BaseFeatureBatch issuance is private")
        batch = object.__new__(cls)
        object.__setattr__(batch, "features", features)
        object.__setattr__(batch, "stock_ids", stock_ids)
        object.__setattr__(batch, "feature_names", feature_names)
        object.__setattr__(batch, "schema_fingerprint", schema_fingerprint)
        object.__setattr__(
            batch,
            "feature_content_fingerprint",
            _tensor_content_fingerprint(features),
        )
        object.__setattr__(batch, "_issued_by_contract", issuance_sentinel)
        batch._validate()
        return batch

    def _validate(self) -> None:
        if self._issued_by_contract is not _ISSUANCE_SENTINEL:
            raise ValueError("BaseFeatureBatch was not issued by BaseFeatureContract")
        if not _SHA256_RE.fullmatch(self.schema_fingerprint):
            raise ValueError("BaseFeatureBatch schema fingerprint must be sha256:<64 hex>")
        if not _SHA256_RE.fullmatch(self.feature_content_fingerprint):
            raise ValueError("BaseFeatureBatch content fingerprint must be sha256:<64 hex>")
        if self.features.ndim != 2:
            raise ValueError("BaseFeatureBatch features must have shape [batch, feature_dim]")
        if self.features.shape[0] != len(self.stock_ids):
            raise ValueError("BaseFeatureBatch stock metadata length must match feature batch size")
        if self.features.shape[1] != len(self.feature_names):
            raise ValueError("BaseFeatureBatch feature_names length must match feature dimension")
        if not torch.isfinite(self.features).all():
            raise ValueError("BaseFeatureBatch features must contain only finite values")


class BaseFeatureContract:
    """Validated Base input feature schema."""

    def __init__(self, feature_specs: Sequence[FeatureSpec]) -> None:
        if not feature_specs:
            raise ValueError("Base feature schema must contain at least one feature")

        names = [spec.name for spec in feature_specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate Base feature names: {duplicates}")

        for spec in feature_specs:
            self._validate_spec(spec)

        self.feature_specs = tuple(feature_specs)
        self.feature_names = tuple(names)
        self.schema_fingerprint = self._fingerprint(self.feature_specs)

    @property
    def input_dim(self) -> int:
        return len(self.feature_specs)

    def validate_tensor(
        self,
        features: torch.Tensor,
        stock_ids: Sequence[str],
        *,
        feature_names: Sequence[str] | None = None,
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("Base feature tensor must have shape [batch, feature_dim]")
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Base feature dimension drift: expected {self.input_dim}, got {features.shape[1]}"
            )
        if features.shape[0] != len(stock_ids):
            raise ValueError("Base stock metadata length must match feature batch size")
        if any(not isinstance(stock_id, str) or not stock_id for stock_id in stock_ids):
            raise ValueError("Base stock metadata must contain non-empty stock IDs")
        if not torch.isfinite(features).all():
            raise ValueError("Base feature tensor must contain only finite values")

        if feature_names is not None:
            names = tuple(feature_names)
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(f"Duplicate Base feature names: {duplicates}")
            if len(names) != self.input_dim:
                raise ValueError("Base feature_names length must match feature dimension")
            if names != self.feature_names:
                unknown = [name for name in names if name not in self.feature_names]
                if unknown:
                    raise ValueError(f"Base feature tensor contains unknown features: {unknown}")
                raise ValueError("Base feature tensor order or names do not match schema")

        return features

    def validate_batch(
        self,
        features: torch.Tensor,
        stock_ids: Sequence[str],
        *,
        feature_names: Sequence[str],
    ) -> BaseFeatureBatch:
        validated = self.validate_tensor(
            features,
            stock_ids,
            feature_names=feature_names,
        )
        owned_features = validated.detach().clone()
        return BaseFeatureBatch._issue(
            issuance_sentinel=_ISSUANCE_SENTINEL,
            features=owned_features,
            stock_ids=tuple(stock_ids),
            feature_names=tuple(feature_names),
            schema_fingerprint=self.schema_fingerprint,
        )

    def revalidate_batch(self, batch: BaseFeatureBatch) -> None:
        if batch._issued_by_contract is not _ISSUANCE_SENTINEL:
            raise ValueError("BaseFeatureBatch was not issued by BaseFeatureContract")
        if batch.schema_fingerprint != self.schema_fingerprint:
            raise ValueError("BaseFeatureBatch schema fingerprint does not match contract")
        self.validate_tensor(
            batch.features,
            batch.stock_ids,
            feature_names=batch.feature_names,
        )
        current_fingerprint = _tensor_content_fingerprint(batch.features)
        if current_fingerprint != batch.feature_content_fingerprint:
            raise ValueError("BaseFeatureBatch content fingerprint changed after validation")

    @staticmethod
    def _validate_spec(spec: FeatureSpec) -> None:
        if is_forbidden_identity_feature_name(spec.name):
            raise ValueError(
                f"Identity leakage detected: identity feature name {spec.name!r}"
            )
        if spec.scope is FeatureScope.IDENTITY or spec.source is SourceKind.IDENTITY:
            raise ValueError(f"Base feature {spec.name!r} is identity metadata, not continuous input")
        if spec.scope is FeatureScope.UNKNOWN or spec.source is SourceKind.UNKNOWN:
            raise ValueError(f"Base feature {spec.name!r} has UNKNOWN scope/source")
        if spec.is_current_event:
            raise ValueError(f"Base feature {spec.name!r} is current-event derived")
        if spec.scope not in _BASE_ALLOWED_SCOPES or spec.source not in _BASE_ALLOWED_SOURCES:
            raise ValueError(
                f"Base feature {spec.name!r} scope/source is not allowed: "
                f"{spec.scope.value}/{spec.source.value}"
            )
        if not spec.pit_required:
            raise ValueError(f"Base feature {spec.name!r} must require PIT semantics")

    @staticmethod
    def _fingerprint(feature_specs: Sequence[FeatureSpec]) -> str:
        payload = [
            {
                "name": spec.name,
                "version": spec.version,
                "horizons": spec.horizons,
                "dependencies": spec.dependencies,
                "scope": spec.scope.value,
                "source": spec.source.value,
                "pit_required": spec.pit_required,
                "is_current_event": spec.is_current_event,
            }
            for spec in feature_specs
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


class BaseMarketModel(nn.Module):
    """MLP baseline using only audited Base feature tensors."""

    def __init__(
        self,
        contract: BaseFeatureContract | None = None,
        input_dim: int | None = None,
        hidden_dim: int = 64,
        num_horizons: int | None = None,
        dropout: float = 0.1,
        _allow_nonformal_legacy: bool = False,
    ) -> None:
        super().__init__()
        if num_horizons is not None and num_horizons != BASE_DAILY_PATH_DAYS:
            raise ValueError("BaseMarketModel output is fixed to five daily increments")
        if contract is None and not _allow_nonformal_legacy:
            raise ValueError("Formal BaseMarketModel construction requires a BaseFeatureContract")
        if contract is not None and input_dim is not None:
            raise ValueError("Base input_dim is derived from BaseFeatureContract")
        if contract is None and input_dim is None:
            raise ValueError("Non-formal legacy BaseMarketModel requires input_dim")

        self.contract = contract
        self.input_dim = contract.input_dim if contract is not None else cast(int, input_dim)
        self.hidden_dim = hidden_dim
        self.is_formal = contract is not None

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, BASE_DAILY_PATH_DAYS),
        )

    @classmethod
    def for_nonformal_legacy(
        cls,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> BaseMarketModel:
        """Build a non-formal adapter for legacy FDG compatibility only."""
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            _allow_nonformal_legacy=True,
        )

    def forward(self, batch: BaseFeatureBatch) -> torch.Tensor:
        """Return formal daily path predictions for a validated Base batch."""
        if not isinstance(batch, BaseFeatureBatch):
            raise TypeError("Formal Base forward requires a BaseFeatureBatch")
        if self.contract is None:
            raise ValueError("Non-formal legacy Base model cannot use formal forward")
        self.contract.revalidate_batch(batch)
        return self._run_network(batch.features)

    def _forward_nonformal_legacy(self, x: torch.Tensor) -> torch.Tensor:
        """Run low-level Base network for explicitly non-formal legacy adapters."""
        if self.is_formal:
            raise ValueError("Formal Base model cannot use non-formal legacy forward")
        return self._run_network(x)

    def _run_network(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Base input tensor must have shape [batch, {self.input_dim}]"
            )
        if not torch.isfinite(x).all():
            raise ValueError("Base input tensor must contain only finite values")
        return cast(torch.Tensor, self.net(x))

    def predict(
        self,
        batch: BaseFeatureBatch,
        *,
        prediction_time: datetime,
        cutoff_time: datetime,
    ) -> tuple[BasePrediction, ...]:
        daily_paths = self.forward(batch)
        return tuple(
            BasePrediction(
                stock_id=stock_id,
                prediction_time=prediction_time,
                cutoff_time=cutoff_time,
                daily_path=cast(
                    tuple[float, float, float, float, float],
                    tuple(float(v) for v in daily_paths[row].detach().cpu().tolist()),
                ),
                feature_schema_fingerprint=batch.schema_fingerprint,
            )
            for row, stock_id in enumerate(batch.stock_ids)
        )

    @property
    def output_dim(self) -> int:
        return BASE_DAILY_PATH_DAYS


class BaseLoss(nn.Module):
    """Combined Huber + rank loss for five-day Base daily paths."""

    def __init__(self, huber_delta: float = 1.0, rank_weight: float = 0.1) -> None:
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.rank_weight = rank_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss over daily paths."""
        _validate_daily_path_tensor(pred)
        _validate_daily_path_tensor(target)
        huber_loss = self.huber(pred, target)
        rank_loss = self._rank_loss(pred, target)
        return cast(torch.Tensor, huber_loss + self.rank_weight * rank_loss)

    @staticmethod
    def _rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pairwise ranking loss: pred order should match target order."""
        if pred.shape[0] < 2:
            return torch.tensor(0.0, device=pred.device)
        p = pred.flatten()
        t = target.flatten()
        n = p.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=pred.device)
        pred_diff = p.unsqueeze(0) - p.unsqueeze(1)
        target_diff = t.unsqueeze(0) - t.unsqueeze(1)
        sign_agreement = torch.sign(pred_diff) * torch.sign(target_diff)
        rank_penalty = torch.clamp(-sign_agreement, min=0.0)
        return rank_penalty.mean()

    def decompose(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """Return decomposed loss components."""
        _validate_daily_path_tensor(pred)
        _validate_daily_path_tensor(target)
        huber_val = self.huber(pred, target).item()
        rank_val = self._rank_loss(pred, target).item()
        return {
            "huber": huber_val,
            "rank": rank_val,
            "total": huber_val + self.rank_weight * rank_val,
        }
