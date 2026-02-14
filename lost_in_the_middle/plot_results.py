#!/usr/bin/env python3
"""
plot_results.py – Visualise Lost-in-the-Middle results.

Usage:
    python -m lost_in_the_middle.plot_results results.jsonl --out plot.png
"""

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POSITION_ORDER = ["start", "early", "middle", "late", "end"]
POSITION_LABELS = {
    "start": "Start\n(pos 1)",
    "early": "Early\n(pos ¼)",
    "middle": "Middle\n(pos ½)",
    "late": "Late\n(pos ¾)",
    "end": "End\n(pos N)",
}


def load_results(path: str):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=str, help="Path to results.jsonl")
    parser.add_argument("--out", type=str, default="plot.png")
    args = parser.parse_args()

    records = load_results(args.results)

    # Group by (context_len, needle_pos)
    groups = defaultdict(list)
    for r in records:
        groups[(r["context_len"], r["needle_pos"])].append(r["correct"])

    context_lens = sorted({r["context_len"] for r in records})
    positions = [p for p in POSITION_ORDER if any((cl, p) in groups for cl in context_lens)]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(positions))
    colors = ["#2196F3", "#FF9800", "#E91E63", "#4CAF50", "#9C27B0"]

    for i, cl in enumerate(context_lens):
        accs = []
        for pos in positions:
            vals = groups.get((cl, pos), [])
            acc = np.mean(vals) * 100 if vals else 0
            accs.append(acc)
        color = colors[i % len(colors)]
        ax.plot(x, accs, "o-", label=f"{cl} documents (~{cl*45} tokens)",
                color=color, linewidth=2, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([POSITION_LABELS.get(p, p) for p in positions])
    ax.set_xlabel("Needle Position in Context", fontsize=12)
    ax.set_ylabel("Exact-Match Accuracy (%)", fontsize=12)
    ax.set_title(
        "Lost in the Middle: Accuracy vs. Needle Position\n"
        "(gpt-4o-mini, synthetic multi-document QA)",
        fontsize=13,
    )
    ax.set_ylim(0, 105)
    ax.legend(title="Context Length", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"✅  Saved plot to {args.out}")


if __name__ == "__main__":
    main()
