"""
将 outputs/train 下所有 TensorBoard 日志导出为 CSV，供 MATLAB 可视化使用。

运行方式：
    lerobot_env\Scripts\python.exe export_tb_to_csv.py

输出目录：outputs/tb_csv/
每个 run 对应一个 CSV 文件，列：
    time_min        : 相对训练时间（分钟）
    step            : Interaction step
    success_rate    : 20 回合滑动平均成功率
    intervention_rate: 20 回合滑动平均干预率
    cycle_time_s    : 20 回合滑动平均周期时间（秒，如无则 NaN）
"""

import os
import glob
import csv
from collections import deque

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ─── 配置 ────────────────────────────────────────────────────────────────────
TRAIN_DIR = r"D:\code\lerobot-hilserl\outputs\train"
EXPORT_DIR = r"D:\code\lerobot-hilserl\outputs\tb_csv"
SLIDE_WINDOW = 20       # 论文：20 回合滑动平均
ENV_FPS = 20            # 用于将 steps 换算为秒（若无 episode duration 数据时备用）
# ─────────────────────────────────────────────────────────────────────────────


def running_avg(values: list, window: int = 20) -> list:
    """计算滑动均值（窗口从 1 增长到 window）。"""
    buf = deque(maxlen=window)
    result = []
    for v in values:
        buf.append(v)
        result.append(sum(buf) / len(buf))
    return result


def align_by_step(primary_events, secondary_events):
    """按 step 对齐两组 events，返回对齐后的 (wall_time, step, p_val, s_val) 列表。"""
    sec_dict = {e.step: e.value for e in secondary_events}
    aligned = []
    for e in primary_events:
        if e.step in sec_dict:
            aligned.append((e.wall_time, e.step, e.value, sec_dict[e.step]))
    return aligned


def export_run(tb_dir: str, run_name: str, export_dir: str):
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))

    reward_tag      = "train/Episodic reward"
    interv_tag      = "train/Intervention rate"
    success_tag     = "train/Episode success"
    duration_tag    = "train/Episode duration (s)"

    # 必须有 reward 或 success
    if reward_tag not in tags and success_tag not in tags:
        print(f"  [SKIP] {run_name}: 无 episode 指标")
        return

    # ── 读取原始数据 ──────────────────────────────────────────────────────────
    reward_events  = ea.Scalars(reward_tag)   if reward_tag in tags  else []
    interv_events  = ea.Scalars(interv_tag)   if interv_tag in tags  else []
    success_events = ea.Scalars(success_tag)  if success_tag in tags else []
    dur_events     = ea.Scalars(duration_tag) if duration_tag in tags else []

    # 以 reward/success 为主轴，对齐 intervention_rate
    if success_events:
        primary = success_events
    else:
        primary = reward_events

    if not primary:
        print(f"  [SKIP] {run_name}: 无有效 episode 数据")
        return

    # 时间原点 = 第一个 episode 的 wall_time
    t0 = primary[0].wall_time

    # 建立 step → wall_time 映射（来自 primary）
    step_to_wt = {e.step: e.wall_time for e in primary}

    # 成功率：用 success 标签；若无则用 reward > 0
    success_vals = {}
    if success_events:
        for e in success_events:
            success_vals[e.step] = float(e.value)
    else:
        for e in reward_events:
            success_vals[e.step] = 1.0 if e.value > 0 else 0.0

    # 干预率
    interv_vals = {e.step: e.value for e in interv_events}

    # 周期时间
    dur_vals = {e.step: e.value for e in dur_events}

    # ── 共同 steps ────────────────────────────────────────────────────────────
    common_steps = sorted(set(success_vals.keys()) & set(interv_vals.keys())
                          if interv_vals
                          else set(success_vals.keys()))

    if not common_steps:
        print(f"  [WARN] {run_name}: step 对齐后无公共数据")
        return

    # ── 计算滑动平均 ──────────────────────────────────────────────────────────
    succ_seq  = [success_vals[s] for s in common_steps]
    interv_seq = [interv_vals.get(s, float("nan")) for s in common_steps]
    dur_seq    = [dur_vals.get(s, float("nan")) for s in common_steps]

    succ_slide  = running_avg(succ_seq,  SLIDE_WINDOW)
    interv_slide = running_avg(
        [v for v in interv_seq if not (v != v)],  # drop NaN
        SLIDE_WINDOW
    ) if any(v == v for v in interv_seq) else [float("nan")] * len(common_steps)

    # 若 interv_slide 长度不一致（因 NaN 过滤），则对齐
    if len(interv_slide) != len(common_steps):
        # 重新计算，保留 NaN 位置
        buf = deque(maxlen=SLIDE_WINDOW)
        interv_slide = []
        for v in interv_seq:
            if v == v:  # not NaN
                buf.append(v)
            interv_slide.append(sum(buf) / len(buf) if buf else float("nan"))

    dur_slide = []
    buf = deque(maxlen=SLIDE_WINDOW)
    for v in dur_seq:
        if v == v:
            buf.append(v)
        dur_slide.append(sum(buf) / len(buf) if buf else float("nan"))

    # ── 写 CSV ────────────────────────────────────────────────────────────────
    out_path = os.path.join(export_dir, f"{run_name}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_min", "step", "success_rate", "intervention_rate", "cycle_time_s"])
        for i, step in enumerate(common_steps):
            wt = step_to_wt.get(step, t0)
            time_min = (wt - t0) / 60.0
            writer.writerow([
                f"{time_min:.4f}",
                step,
                f"{succ_slide[i]:.6f}",
                f"{interv_slide[i]:.6f}",
                f"{dur_slide[i]:.6f}" if dur_slide[i] == dur_slide[i] else "NaN",
            ])

    print(f"  [OK] {run_name}: {len(common_steps)} episodes → {out_path}")


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # 查找所有 tensorboard/default 目录
    pattern = os.path.join(TRAIN_DIR, "*", "*", "tensorboard", "default")
    tb_dirs = sorted(glob.glob(pattern))

    if not tb_dirs:
        print(f"未找到任何 TensorBoard 目录: {pattern}")
        return

    print(f"找到 {len(tb_dirs)} 个 run，开始导出...\n")
    for tb_dir in tb_dirs:
        parts = tb_dir.replace("\\", "/").split("/")
        # 路径结构: .../outputs/train/DATE/TIME_name/tensorboard/default
        # 倒数: parts[-1]=default, [-2]=tensorboard, [-3]=TIME_name, [-4]=DATE
        date_str = parts[-4]   # 2026-03-13
        time_str = parts[-3]   # 13-58-17_default
        run_name = f"{date_str}_{time_str}"
        print(f"处理: {run_name}")
        try:
            export_run(tb_dir, run_name, EXPORT_DIR)
        except Exception as e:
            print(f"  [ERROR] {run_name}: {e}")

    print(f"\n导出完成，CSV 文件位于: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
