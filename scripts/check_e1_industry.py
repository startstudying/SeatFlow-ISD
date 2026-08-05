"""Quick check: does Phase 2 industry wiring create industry nodes in the
graph, and does E1 differ from E0 with the fix? Uses a tiny 20-day window
so it fits one foreground run.
"""
import logging
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from seatflow.evaluation.r5_real_runner_v3 import run_r5_real_v3_experiment
from seatflow.experiment.matrix import get_experiment_matrix

matrix = get_experiment_matrix()

out = Path("results/r5/e01_real/E1_industry_check")
out.mkdir(parents=True, exist_ok=True)

# E0 and E1 on the same tiny window
for name in ("E0", "E1"):
    cfg = matrix.get_by_name(name)
    res = run_r5_real_v3_experiment(
        experiment_config=cfg,
        data_root=Path("data/historical_backfill/tushare_2023_2026_v2"),
        output_root=out / name,
        seeds=(2026,),
        min_train_dates=2,
        purge_days=0,
        embargo_days=0,
        n_folds=1,
        epochs_stage1=1,
        epochs_stage2=1,
        epochs_stage3=1,
        as_of_start=date(2023, 6, 1),
        as_of_end=date(2023, 6, 21),
        observed_label_end=date(2023, 6, 21),
        chunk_dates=15,
        cooccurrence_cap=5,
    )
    rows = res.to_dict().get("rows", [])
    for r in rows:
        rank_ic = r["rank_ic"]
        rank_ic_str = f"{rank_ic:.4f}" if rank_ic is not None else "None"
        print(f"  {name} seed={r['seed']} fold={r['fold_id']} rank_ic={rank_ic_str} source={r['metrics_source']}")
