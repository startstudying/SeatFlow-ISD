"""Materialize an auditable lightweight matrix from the frozen R1 tensor dataset."""

from __future__ import annotations

import argparse
import json
import pickle
from hashlib import sha256
from pathlib import Path

import numpy as np

from seatflow.datasets.pyg import (
    EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA,
    TensorizedWindowSample,
)
from seatflow.evaluation import dt_baselines


def main() -> None:
    args = _arguments()
    source_manifest_path = args.dataset_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("label_schema_version") != EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA:
        raise ValueError("R1 matrix cache requires the executable V2 label schema")

    dataset_path = args.dataset_root / "v2-tensor-dataset.pkl"
    if _file_sha256(dataset_path) != source_manifest.get("dataset_sha256"):
        raise ValueError("R1 source dataset hash does not match its manifest")
    samples = pickle.loads(dataset_path.read_bytes())
    if not isinstance(samples, tuple) or any(
        not isinstance(sample, TensorizedWindowSample) for sample in samples
    ):
        raise ValueError("R1 source dataset is not a tuple of tensorized samples")
    if any(
        sample.label_schema_version != EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA
        for sample in samples
    ):
        raise ValueError("R1 matrix cache rejects mixed label schemas")

    values, feature_names = _matrix(samples)
    sample_ids = np.asarray([sample.sample_id for sample in samples])
    as_of_dates = np.asarray(
        [sample.as_of_date.isoformat() if sample.as_of_date else "" for sample in samples]
    )
    labels = np.asarray([sample.labels[1] for sample in samples], dtype=np.int8)
    excess_returns = np.asarray([sample.returns[1] for sample in samples], dtype=np.float64)
    if any(not value for value in as_of_dates):
        raise ValueError("R1 matrix cache requires every sample to have an as-of date")

    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / "matrix.npz"
    np.savez_compressed(
        matrix_path,
        x=values,
        sample_ids=sample_ids,
        as_of_dates=as_of_dates,
        labels=labels,
        excess_returns=excess_returns,
        feature_names=np.asarray(feature_names),
    )
    payload = {
        "status": "materialized_fail_closed",
        "model_valid": False,
        "label_schema_version": EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_manifest["sha256"],
        "source_dataset_sha256": source_manifest["dataset_sha256"],
        "sample_count": len(samples),
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "event_amount_semantics": "inverse_signed_log_before_aggregation",
        "sample_ids_sha256": _hash_json(sample_ids.tolist()),
        "matrix": str(matrix_path),
        "matrix_sha256": _file_sha256(matrix_path),
        "feature_extractor_sha256": _file_sha256(
            Path(str(dt_baselines.__file__))
        ),
        "code_sha256": _file_sha256(Path(__file__)),
    }
    payload["sha256"] = _hash_json(payload)
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "samples": payload["sample_count"],
                "features": payload["feature_count"],
                "matrix_sha256": payload["matrix_sha256"],
            }
        )
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _matrix(
    samples: tuple[TensorizedWindowSample, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    names: tuple[str, ...] | None = None
    rows: list[tuple[float, ...]] = []
    for sample in samples:
        stock_names = sample.graph.metadata.get("feature_schema", {}).get("stock", [])
        vector = dt_baselines._feature_vector(sample) + tuple(
            float(value)
            for value in sample.graph.data["stock"]
            .x[sample.target_node_index]
            .detach()
            .cpu()
            .tolist()
        )
        current = tuple(
            f"event_profile::{index}" for index in range(len(vector) - len(stock_names))
        ) + tuple(f"stock::{name}" for name in stock_names)
        if names is None:
            names = current
        elif names != current:
            raise ValueError("R1 tensor feature schema is not stable")
        rows.append(vector)
    return np.asarray(rows, dtype=np.float64), names or ()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    main()
