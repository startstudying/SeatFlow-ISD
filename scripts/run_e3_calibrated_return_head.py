"""Run the bounded D3/E3 return-calibration diagnostic on verified snapshots."""

from seatflow.cli import main

if __name__ == "__main__":
    main(
        [
            "dt-rolling-run",
            "--data-root",
            "data/historical_backfill/tushare_2023_2026_v2",
            "--output-root",
            "artifacts/diagnostics/E3_CALIBRATED_RETURN_HEAD",
            "--result-file",
            "artifacts/diagnostics/E3_CALIBRATED_RETURN_HEAD/result.json",
            "--as-of-start",
            "2023-02-06",
            "--as-of-end",
            "2023-02-28",
            "--observed-label-end",
            "2023-03-10",
            "--diagnostic-e3",
            "--seed",
            "2026",
            "--epochs",
            "3",
            "--min-train-dates",
            "8",
            "--chunk-dates",
            "5",
        ]
    )
