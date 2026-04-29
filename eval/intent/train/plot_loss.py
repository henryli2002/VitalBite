"""Parse mlx_lm training logs and plot loss curves for all experiments."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG_DIR = Path(__file__).parent / "experiments" / "logs"
OUT_DIR = Path(__file__).parent / "experiments"

TRAIN_RE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+)")
VALID_RE  = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")

EXP_LABELS = {
    "exp_A_all_8layers":  "A: all, 8 layers",
    "exp_B_attn_8layers": "B: attn, 8 layers",
    "exp_C_mlp_8layers":  "C: mlp, 8 layers",
    "exp_D_all_16layers": "D: all, 16 layers",
}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def parse_log(log_path: Path):
    train, valid = [], []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if m := TRAIN_RE.search(line):
            train.append((int(m.group(1)), float(m.group(2))))
        if m := VALID_RE.search(line):
            valid.append((int(m.group(1)), float(m.group(2))))
    return train, valid


def main():
    logs = sorted(LOG_DIR.glob("exp_*.log"))
    if not logs:
        print(f"No logs found in {LOG_DIR}", file=sys.stderr)
        sys.exit(1)

    fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Qwen3.5-0.8B LoRA Experiments — Training & Validation Loss", fontsize=13)

    for idx, log_path in enumerate(logs):
        name = log_path.stem
        label = EXP_LABELS.get(name, name)
        color = COLORS[idx % len(COLORS)]
        train, valid = parse_log(log_path)

        if train:
            xs, ys = zip(*train)
            ax_t.plot(xs, ys, label=label, color=color, linewidth=1.5)
        if valid:
            xs, ys = zip(*valid)
            ax_v.plot(xs, ys, label=label, color=color, linewidth=1.5, marker="o", markersize=4)

    for ax, title in [(ax_t, "Train Loss"), (ax_v, "Val Loss")]:
        ax.set_title(title)
        ax.set_xlabel("Iter")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "loss_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
