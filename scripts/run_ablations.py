"""
3 项消融实验脚本 — 补跑 seat_profile / time_edges / direction_edges。

调用 run_dt_seathgt_rolling_evaluation 并指定 ablation group。

运行方式:
    python scripts/run_ablations.py --group seat_profile
    python scripts/run_ablations.py --group time_edges
    python scripts/run_ablations.py --group direction_edges
"""

from pathlib import Path
import argparse
import json
from datetime import datetime

ABLATION_OUTPUT_ROOT = "artifacts/ablations"

def run_ablation(group: str, seeds: tuple[int, ...] = (2025, 2026, 2027)):
    """运行单个消融实验。"""

    if group not in ("seat_profile", "time_edges", "direction_edges"):
        raise ValueError(f"不支持的消融组: {group}")

    output_dir = Path(ABLATION_OUTPUT_ROOT) / group
    output_dir.mkdir(parents=True, exist_ok=True)

    # 调用实际的 rolling evaluation 入口
    # 这里是框架脚本,具体命令需根据实际入口点调整
    from seatflow.evaluation.dt_rolling import run_dt_seathgt_rolling_evaluation
    from seatflow.data import load_verified_historical_snapshots
    from seatflow.training.dt_seat_hgt_v3_training import DTSeatHGTTrainingConfig

    print(f"运行 {group} 消融实验...")

    # 加载样本和配置
    samples = load_verified_historical_snapshots(
        # 实际参数需根据数据路径调整
    )

    config = DTSeatHGTTrainingConfig()

    # 执行消融实验
    result_path = run_dt_seathgt_rolling_evaluation(
        samples,
        config,
        output_dir,
        selection_horizon=1,
        seeds=seeds,
        n_folds=3,
        ablation_group=group,
    )

    # 读取结果
    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)

    # 提取消融指标
    ablation_rows = result.get("ablations", [])
    if ablation_rows:
        ablation = ablation_rows[0]
        auc_contribution = ablation.get("auc_contribution", 0.0)
        rank_ic_contribution = ablation.get("rank_ic_contribution", 0.0)

        print(f"{group} 消融实验完成:")
        print(f"  AUC 贡献: {auc_contribution:.4f}")
        print(f"  RankIC 贡献: {rank_ic_contribution:.4f}")
    else:
        print(f"{group} 消融实验结果未找到")

    return result_path

def main():
    parser = argparse.ArgumentParser(description="运行 DT-SeatHGT 消融实验")
    parser.add_argument("--group", required=True, help="消融组: seat_profile / time_edges / direction_edges")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2025, 2026, 2027], help="随机种子")

    args = parser.parse_args()

    result_path = run_ablation(args.group, tuple(args.seeds))

    print(f"消融实验结果已写入: {result_path}")

if __name__ == "__main__":
    main()