"""
R5.4 行业重跑脚本 — 用 PR #10 合入的 sw2021 L1 industry membership 重跑 R5.4 验证 E1 > E0。

运行方式:
    python scripts/r54_industry_rerun.py
"""

from pathlib import Path
import subprocess
import json

INDUSTRY_MEMBERSHIP_ROOT = "data/historical_backfill/tushare_2023_2026_v2/r1_industry_membership"
R54_OUTPUT_ROOT = "artifacts/r5/r54_industry_rerun"

def main():
    root = Path(INDUSTRY_MEMBERSHIP_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"行业数据不存在: {root}")

    output_root = Path(R54_OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    # 调用 load_verified_historical_snaphots 时传入 industry_membership_root
    # 具体命令需根据实际入口点调整 — 这里是执行框架脚本
    cmd = [
        "python", "-m", "seatflow", "research",
        "r54_industry_rerun",
        f"--industry-membership-root={INDUSTRY_MEMBERSHIP_ROOT}",
        f"--output-root={R54_OUTPUT_ROOT}",
    ]

    print(f"执行 R5.4 行业重跑...")
    print(f"命令: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"R5.4 行业重跑失败,退出码: {result.returncode}")
        return

    print(f"R5.4 行业重跑完成,输出: {R54_OUTPUT_ROOT}")

    # 验证 E1 > E0
    status_file = output_root / "r5_status.json"
    if status_file.exists():
        with open(status_file, encoding="utf-8") as f:
            status = json.load(f)

        # 查找 E1 vs E0 对比结果
        e0_rank_ic = status.get("e0", {}).get("mean_rank_ic")
        e1_rank_ic = status.get("e1", {}).get("mean_rank_ic")

        print(f"E0 mean_rank_ic: {e0_rank_ic}")
        print(f"E1 mean_rank_ic: {e1_rank_ic}")

        if e1_rank_ic is not None and e0_rank_ic is not None:
            improvement = e1_rank_ic - e0_rank_ic
            print(f"E1 vs E0 提升: {improvement:.4f}")

            if improvement > 0:
                print("✓ E1 > E0 验证通过")
            else:
                print("✗ E1 > E0 验证失败")
        else:
            print("未找到 E0/E1 结果")

if __name__ == "__main__":
    main()