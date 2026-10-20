#!/usr/bin/env python3
"""可视化 JointRecorder 导出的 csv：策略期望位置 q_des vs 实测位置 q（单张图）。"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# SDK 电机顺序（与 config.yaml joint_names 一致）
DEFAULT_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

RAD2DEG = 180.0 / np.pi


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_logs = here.parent / "logs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=None,
        help="csv 路径；默认取 logs/ 下最新的 *.csv",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=default_logs,
        help=f"默认日志目录 (default: {default_logs})",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        default=None,
        help="只画指定关节名（可部分匹配，如 left_knee hip_pitch）",
    )
    parser.add_argument("--deg", action="store_true", help="纵轴用 deg（默认 rad）")
    parser.add_argument("--error", action="store_true", help="额外绘制跟踪误差 q_des - q")
    parser.add_argument(
        "--max-points",
        type=int,
        default=3000,
        help="绘图最大点数，过大则均匀降采样 (default: 3000)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="保存图片路径；默认与 csv 同名的 .png",
    )
    parser.add_argument("--show", action="store_true", help="弹窗显示（默认只保存）")
    parser.add_argument("--ncols", type=int, default=5, help="子图列数 (default: 5)")
    return parser.parse_args()


def find_latest_csv(logs_dir: Path) -> Path:
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"日志目录不存在: {logs_dir}")
    cands = sorted(logs_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"{logs_dir} 下没有 csv")
    return cands[0]


def load_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    data = {name: np.empty(len(rows), dtype=np.float64) for name in header}
    for i, row in enumerate(rows):
        for j, name in enumerate(header):
            data[name][i] = float(row[j])
    return header, data


def discover_joints(header: list[str]) -> list[str]:
    found = [name[: -len("_q_des")] for name in header if name.endswith("_q_des")]
    # 优先按 SDK 顺序排
    ordered = [j for j in DEFAULT_JOINT_ORDER if j in found]
    ordered += [j for j in found if j not in ordered]
    return ordered


def select_joints(all_joints: list[str], patterns: list[str] | None) -> list[str]:
    if not patterns:
        return all_joints
    selected = [j for j in all_joints if any(re.search(pat, j) for pat in patterns)]
    if not selected:
        raise ValueError(f"--joints {patterns} 未匹配到任何关节，可选: {all_joints}")
    return selected


def downsample(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).astype(int)


def plot_all(
    t: np.ndarray,
    data: dict[str, np.ndarray],
    joints: list[str],
    *,
    title: str,
    use_deg: bool,
    show_error: bool,
    ncols: int,
    out: Path | None,
    show: bool,
) -> None:
    scale = RAD2DEG if use_deg else 1.0
    unit = "deg" if use_deg else "rad"
    n = len(joints)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.1 * nrows + 0.6),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(title, fontsize=11)

    for idx, joint in enumerate(joints):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        q_des = data[f"{joint}_q_des"] * scale
        q = data[f"{joint}_q"] * scale
        ax.plot(t, q_des, label="q_des", lw=0.9, color="C0")
        ax.plot(t, q, label="q", lw=0.9, color="C1", alpha=0.85)
        if show_error:
            ax2 = ax.twinx()
            ax2.plot(t, q_des - q, lw=0.7, color="C3", alpha=0.45)
            ax2.tick_params(axis="y", labelsize=6, labelcolor="C3")
        ax.set_title(joint.replace("_joint", ""), fontsize=8)
        ax.set_ylabel(unit, fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=6, framealpha=0.7)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    for c in range(ncols):
        axes[-1][c].set_xlabel("t [s]", fontsize=7)

    fig.tight_layout()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"saved: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv or find_latest_csv(args.logs_dir)
    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    print(f"loading: {csv_path}")
    header, data = load_csv(csv_path)
    joints = select_joints(discover_joints(header), args.joints)
    print(f"samples: {len(data['t'])}, joints: {len(joints)}")

    idx = downsample(len(data["t"]), args.max_points)
    t = data["t"][idx]
    slim = {k: v[idx] for k, v in data.items()}

    out = args.out or csv_path.with_suffix(".png")
    plot_all(
        t,
        slim,
        joints,
        title=f"{csv_path.name}  |  q_des vs q",
        use_deg=args.deg,
        show_error=args.error,
        ncols=args.ncols,
        out=out,
        show=args.show,
    )


if __name__ == "__main__":
    main()
