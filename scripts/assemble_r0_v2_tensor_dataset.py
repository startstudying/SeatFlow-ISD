"""Assemble independently verified V2 tensor chunks without weakening their provenance."""

from __future__ import annotations

import argparse
import json
import pickle
from hashlib import sha256
from pathlib import Path

from seatflow.datasets.pyg import EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA, TensorizedWindowSample


def main() -> None:
    args = _arguments()
    manifests = tuple(sorted(Path(args.root).glob("R0_V2_TENSORS_*/manifest.json")))
    if not manifests:
        raise ValueError("no V2 tensor chunk manifests found")
    samples: list[TensorizedWindowSample] = []
    sources: list[dict[str, object]] = []
    source_dates: set[str] = set()
    zero_sample_dates: list[str] = []
    ids: set[str] = set()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "materialized_fail_closed"
            or manifest.get("model_valid") is not False
        ):
            raise ValueError(f"chunk manifest is not fail-closed: {manifest_path}")
        if manifest.get("label_schema_version") != EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA:
            raise ValueError(f"chunk manifest uses a non-V2 schema: {manifest_path}")
        chunk = Path(str(manifest["tensor_chunk"]))
        if sha256(chunk.read_bytes()).hexdigest() != manifest.get("tensor_chunk_sha256"):
            raise ValueError(f"chunk hash mismatch: {chunk}")
        values = pickle.loads(chunk.read_bytes())
        if not isinstance(values, tuple) or any(
            not isinstance(item, TensorizedWindowSample) for item in values
        ):
            raise ValueError(f"invalid tensor chunk payload: {chunk}")
        label_artifact = Path(str(manifest["label_artifact_path"]))
        label_payload = json.loads(label_artifact.read_text(encoding="utf-8"))
        as_of_date = str(label_payload.get("as_of_date", ""))
        if not as_of_date:
            raise ValueError(f"label artifact lacks as_of_date: {label_artifact}")
        source_dates.add(as_of_date)
        for sample in values:
            if sample.label_schema_version != EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA:
                raise ValueError("combined V2 dataset cannot include another label schema")
            if sample.sample_id in ids:
                raise ValueError(f"duplicate sample id across chunks: {sample.sample_id}")
            ids.add(sample.sample_id)
            samples.append(sample)
        sources.append(
            {
                "manifest": str(manifest_path),
                "sha256": manifest["sha256"],
                "as_of_date": as_of_date,
                "tensor_sample_count": len(values),
            }
        )
        if not values:
            zero_sample_dates.append(as_of_date)
    ordered = tuple(sorted(samples, key=lambda item: (item.as_of_date, item.sample_id)))
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    dataset = output / "v2-tensor-dataset.pkl"
    dataset.write_bytes(pickle.dumps(ordered, protocol=pickle.HIGHEST_PROTOCOL))
    payload = {
        "status": "assembled_fail_closed",
        "model_valid": False,
        "label_schema_version": EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA,
        "source_chunks": sources,
        "source_as_of_dates": sorted(source_dates),
        "training_as_of_dates": sorted(
            {item.as_of_date.isoformat() for item in ordered if item.as_of_date}
        ),
        "zero_sample_as_of_dates": sorted(zero_sample_dates),
        "sample_count": len(ordered),
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset.read_bytes()).hexdigest(),
    }
    payload["sha256"] = _hash(payload)
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "samples": len(ordered),
                "source_dates": len(payload["source_as_of_dates"]),
                "training_dates": len(payload["training_as_of_dates"]),
                "zero_sample_dates": len(payload["zero_sample_as_of_dates"]),
            }
        )
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/diagnostics")
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def _hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
