"""
HIL-SERL 训练曲线可视化 —— 论文 Figure 25 风格
双 Y 轴: 左轴 = 成功率 + 干预率，右轴 = 周期时间（若有）
X 轴: 训练时间（分钟）

运行:
    python plot_hil_curves.py
或指定要对比的 run:
    python plot_hil_curves.py --runs 2026-03-13_13-58-17_default 2026-03-16_18-00-22_default
    python plot_hil_curves.py --mode subplots   # 每个 run 一张子图，对应论文多面板
"""

import os
import sys
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ─── 配置 ────────────────────────────────────────────────────────────────────
CSV_DIR   = r"D:\code\lerobot-hilserl\outputs\tb_csv"
OUT_FILE  = os.path.join(CSV_DIR, "hil_learning_curves.png")
DPI       = 200

# 自定义 run 标签（文件名 → 显示名），不在此字典中的 run 用文件名显示
CUSTOM_LABELS = {
    "2026-03-08_21-28-47_default": "Mar-08",
    "2026-03-09_13-34-12_default": "Mar-09",
    "2026-03-11_10-30-34_default": "Mar-11",
    "2026-03-12_18-18-26_default": "Mar-12",
    "2026-03-13_10-21-33_default": "Mar-13 v1",
    "2026-03-13_13-58-17_default": "Mar-13 v2",
    "2026-03-16_18-00-22_default": "Mar-16",
    "2026-03-17_20-12-39_default": "Mar-17",
    "2026-03-19_10-38-14_default": "Mar-19 v1",
    "2026-03-19_11-47-40_default": "Mar-19 v2",
}
# ─────────────────────────────────────────────────────────────────────────────

# 每个 run 的颜色（按顺序循环）
PALETTE = [
    "#E64B35",  # 红
    "#4DBBD5",  # 蓝
    "#00A087",  # 绿
    "#F39B7F",  # 橙
    "#8491B4",  # 蓝灰
    "#91D1C2",  # 浅绿
    "#DC0000",  # 深红
    "#3C5488",  # 深蓝
    "#B09C85",  # 棕
    "#7E6148",  # 深棕
]

SUCCESS_ALPHA    = 1.0
INTERV_ALPHA     = 0.65
CYCLETIME_ALPHA  = 0.75
LINE_WIDTH       = 1.8

# 平滑参数：对应论文里的视觉平滑程度
# sigma 越大越平滑。论文效果约 sigma=15~25（按 episode 索引）
SMOOTH_SIGMA     = 2


def gaussian_smooth(y: "np.ndarray", sigma: float) -> "np.ndarray":
    """对信号做 Gaussian 平滑（sigma 为标准差，单位：索引数）。"""
    import numpy as np
    if sigma <= 0:
        return y
    kernel_half = int(4 * sigma)
    x = np.arange(-kernel_half, kernel_half + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    # 边缘用 edge 值填充，避免边界失真
    y_pad = np.pad(y, kernel_half, mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")[: len(y)]


def load_csv(csv_dir: str, run_names: list[str]) -> dict[str, pd.DataFrame]:
    data = {}
    for name in run_names:
        path = os.path.join(csv_dir, f"{name}.csv")
        if not os.path.isfile(path):
            print(f"[WARN] 找不到: {path}")
            continue
        df = pd.read_csv(path)
        df["cycle_time_s"] = pd.to_numeric(df["cycle_time_s"], errors="coerce")
        data[name] = df
        print(f"[OK] {name}: {len(df)} episodes, 最大 {df['time_min'].max():.1f} min")
    return data


def label_of(name: str) -> str:
    return CUSTOM_LABELS.get(name, name)


def plot_combined(data: dict[str, pd.DataFrame], out_file: str, sigma: float = SMOOTH_SIGMA):
    """所有 run 画在同一张图（综合对比），论文双 Y 轴风格。"""
    import numpy as np
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    has_cycle = False
    legend_handles = []

    for idx, (name, df) in enumerate(data.items()):
        color = PALETTE[idx % len(PALETTE)]
        lbl   = label_of(name)
        t     = df["time_min"].values
        succ  = gaussian_smooth(df["success_rate"].values, sigma)
        intrv = gaussian_smooth(df["intervention_rate"].values, sigma)
        ct    = df["cycle_time_s"].values

        # 成功率（实线）
        l1, = ax1.plot(t, succ,  color=color, lw=LINE_WIDTH,
                       alpha=SUCCESS_ALPHA, linestyle="-",
                       label=f"{lbl} – Success Rate")
        # 干预率（虚线，稍淡）
        l2, = ax1.plot(t, intrv, color=color, lw=LINE_WIDTH * 0.85,
                       alpha=INTERV_ALPHA,  linestyle="--",
                       label=f"{lbl} – Intervention Rate")
        # 周期时间（点划线，右轴）
        ct_mask = ~pd.isna(ct)
        if ct_mask.any():
            has_cycle = True
            ct_smooth = gaussian_smooth(ct[ct_mask], sigma)
            l3, = ax2.plot(t[ct_mask], ct_smooth, color=color,
                           lw=LINE_WIDTH * 0.85, alpha=CYCLETIME_ALPHA,
                           linestyle=":", label=f"{lbl} – Cycle Time (s)")
            legend_handles.append(l3)

        legend_handles.extend([l1, l2])

    # ── 格式 ────────────────────────────────────────────────────────────────
    ax1.set_xlabel("Training Time (Minutes)", fontsize=12)
    ax1.set_ylabel("Success Rate & Intervention Rate", fontsize=11)
    ax1.set_ylim(-0.02, 1.08)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.tick_params(labelsize=10)

    if has_cycle:
        ax2.set_ylabel("Cycle Time (s)", fontsize=11, color="#2c7a2c")
        ax2.tick_params(colors="#2c7a2c", labelsize=10)
        ax2.spines["right"].set_color("#2c7a2c")
    else:
        ax2.set_visible(False)
        ax2.spines["right"].set_visible(False)

    ax1.spines["right"].set_visible(False)

    # 图例
    ax1.legend(handles=legend_handles, loc="center right",
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(1.0 if not has_cycle else 0.88, 0.5))

    fig.suptitle("HIL-SERL Learning Curves  (20-episode running average)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    fig.savefig(out_file, dpi=DPI, bbox_inches="tight")
    print(f"\n[SAVED] {out_file}")
    plt.show()


def plot_subplots(data: dict[str, pd.DataFrame], out_file: str, sigma: float = SMOOTH_SIGMA):
    """每个 run 一个子图，还原论文多面板风格（Figure 25）。"""
    import numpy as np
    n = len(data)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.8),
                             constrained_layout=True)
    axes_flat = [axes] if n == 1 else list(axes.flatten() if nrows > 1 else axes)

    for idx, (name, df) in enumerate(data.items()):
        ax1 = axes_flat[idx]
        ax2 = ax1.twinx()
        color_succ  = "#E24C3F"   # 粉红 = 成功率
        color_intrv = "#4B7CC0"   # 蓝   = 干预率
        color_ct    = "#3CA870"   # 绿   = 周期时间

        t     = df["time_min"].values
        succ  = gaussian_smooth(df["success_rate"].values, sigma)
        intrv = gaussian_smooth(df["intervention_rate"].values, sigma)
        ct    = df["cycle_time_s"].values

        ax1.plot(t, succ,  color=color_succ,  lw=LINE_WIDTH, label="Success Rate")
        ax1.plot(t, intrv, color=color_intrv, lw=LINE_WIDTH, label="Intervention Rate")

        ct_mask = ~pd.isna(ct)
        has_ct  = ct_mask.any()
        if has_ct:
            ct_smooth = gaussian_smooth(ct[ct_mask], sigma)
            ax2.plot(t[ct_mask], ct_smooth, color=color_ct,
                     lw=LINE_WIDTH, label="Cycle Time (s)")
            ax2.set_ylabel("Cycle Time (s)", color=color_ct, fontsize=9)
            ax2.tick_params(colors=color_ct, labelsize=9)
        else:
            ax2.set_visible(False)

        ax1.set_ylim(-0.02, 1.08)
        ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax1.set_xlabel("Training Time (Minutes)", fontsize=9)
        ax1.set_ylabel("Success Rate & Intervention Rate", fontsize=9)
        ax1.set_title(label_of(name), fontsize=10, fontweight="bold")
        ax1.grid(True, alpha=0.25, linestyle="--")
        ax1.tick_params(labelsize=9)
        ax1.spines["top"].set_visible(False)
        if not has_ct:
            ax1.spines["right"].set_visible(False)

        # 图例（与论文同侧）
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2,
                   loc="center right", fontsize=8, framealpha=0.85)

    # 隐藏多余子图
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("HIL-SERL Learning Curves  (20-episode running average)",
                 fontsize=13, fontweight="bold")

    fig.savefig(out_file, dpi=DPI, bbox_inches="tight")
    print(f"\n[SAVED] {out_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",  nargs="*", default=None,
                        help="指定要显示的 run（文件名，不含 .csv），默认全部")
    parser.add_argument("--mode",  choices=["combined", "subplots"], default="combined",
                        help="combined=综合对比图, subplots=每 run 一个子图（论文多面板）")
    parser.add_argument("--out",   default=OUT_FILE, help="输出图片路径")
    parser.add_argument("--no-show",     action="store_true", help="不弹出预览窗口，只保存图片")
    parser.add_argument("--smooth-sigma", type=float, default=SMOOTH_SIGMA,
                        help=f"Gaussian 平滑标准差（episode 数），0=不平滑，默认 {SMOOTH_SIGMA}")
    args = parser.parse_args()

    # 发现 run
    if args.runs:
        run_names = args.runs
    else:
        csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
        run_names = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

    if not run_names:
        print(f"未找到 CSV 文件: {CSV_DIR}")
        sys.exit(1)

    print(f"加载 {len(run_names)} 个 run...\n")
    data = load_csv(CSV_DIR, run_names)
    if not data:
        print("无有效数据")
        sys.exit(1)

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    sigma = args.smooth_sigma
    if args.mode == "subplots":
        plot_subplots(data, args.out, sigma=sigma)
    else:
        plot_combined(data, args.out, sigma=sigma)


if __name__ == "__main__":
    main()
