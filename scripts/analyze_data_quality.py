"""Data quality analysis for Section 4.1 of the report."""

import json
import math
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib / numpy not found.")
    sys.exit(1)

# ── constants ────────────────────────────────────────────────────────────────
ANCHOR_X, ANCHOR_Y = 0.40, 0.10
PLATE_POS           = (0.50, -0.40)
EVAL_JITTER         = 0.05
ROBOT_BASE          = (0.35, -0.74)
R_MAX_H             = 0.845   # Franka horizontal reach at grasp height
R_MIN_H             = 0.20

DATA_DIR = Path(__file__).parent.parent / "data" / "AI-final-49"
OUT_DIR  = Path(__file__).parent.parent / "data" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
def to_world(tvec):
    return ANCHOR_X + tvec[0], ANCHOR_Y + tvec[1]

def in_workspace(wx, wy):
    d = math.sqrt((wx - ROBOT_BASE[0])**2 + (wy - ROBOT_BASE[1])**2)
    return R_MIN_H <= d <= R_MAX_H

def load_positions(path):
    with open(path) as f:
        data = json.load(f)
    result = []
    for ep in data:
        if ep.get("status") != "full":
            continue
        pos = {}
        for obj in ep["objects"]:
            if obj["object_name"] == "plate":
                continue
            pos[obj["object_name"]] = to_world(obj["tvec"])
        source = "synthetic" if ep["video_name"].startswith("synthetic") else "real"
        result.append({"positions": pos, "source": source})
    return result

def workspace_arc(cx, cy, r, n=360):
    thetas = np.linspace(0, 2 * math.pi, n)
    return cx + r * np.cos(thetas), cy + r * np.sin(thetas)

# ── Figure 1: Data pipeline ──────────────────────────────────────────────────
def fig_pipeline():
    with open(DATA_DIR / "object_poses_combined.json") as f:
        combined = json.load(f)
    real_n = sum(1 for e in combined if not e["video_name"].startswith("synthetic"))
    syn_n  = sum(1 for e in combined if e["video_name"].startswith("synthetic"))
    base_n = real_n + syn_n

    stages = ["UMI\nRecorded\n(raw)", "UMI\nIn Workspace\n(filtered)",
              "Base Episodes\n(real + synthetic)", "Datagen\nEpisodes\n(×10 augment)"]
    counts = [49, real_n, base_n, base_n * 10]
    colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(stages, counts, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(count), ha="center", va="bottom", fontweight="bold", fontsize=13)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_ylabel("Number of Episodes", fontsize=11)
    ax.set_title("Data Pipeline: UMI Recording → Datagen Episodes", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / "fig1_data_pipeline.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── Figure 2: Spatial coverage ───────────────────────────────────────────────
def fig_spatial_coverage():
    combined = load_positions(DATA_DIR / "object_poses_combined.json")

    fork_real_x, fork_real_y = [], []
    fork_syn_x,  fork_syn_y  = [], []
    knife_real_x, knife_real_y = [], []
    knife_syn_x,  knife_syn_y  = [], []

    for ep in combined:
        src, pos = ep["source"], ep["positions"]
        if "fork"  in pos:
            (fork_real_x  if src == "real" else fork_syn_x ).append(pos["fork"][0])
            (fork_real_y  if src == "real" else fork_syn_y ).append(pos["fork"][1])
        if "knife" in pos:
            (knife_real_x if src == "real" else knife_syn_x).append(pos["knife"][0])
            (knife_real_y if src == "real" else knife_syn_y).append(pos["knife"][1])

    fig, ax = plt.subplots(figsize=(7, 7))

    # workspace boundary (annulus)
    ox, oy = workspace_arc(*ROBOT_BASE, R_MAX_H)
    ix, iy = workspace_arc(*ROBOT_BASE, R_MIN_H)
    ax.fill(ox, oy, color="#e3f2fd", alpha=0.5, zorder=0)
    ax.plot(ox, oy, color="#1565C0", lw=1.5, linestyle="--", label="Franka workspace boundary")
    ax.fill(ix, iy, color="white",   alpha=1.0, zorder=1)

    # eval range boxes
    for label, (dx, dy), col in [("Eval range — fork",  (0.55, -0.10), "#2196F3"),
                                   ("Eval range — knife", (0.50, -0.10), "#FF5722")]:
        ax.add_patch(plt.Rectangle((dx - EVAL_JITTER, dy - EVAL_JITTER),
                                   2*EVAL_JITTER, 2*EVAL_JITTER,
                                   lw=1.5, edgecolor=col, facecolor=col, alpha=0.15,
                                   linestyle="--", label=label))
        ax.plot(dx, dy, "*", ms=11, color=col, zorder=5)

    # data points
    ax.scatter(fork_syn_x,   fork_syn_y,   c="#90CAF9", s=30, alpha=0.6, marker="^")
    ax.scatter(fork_real_x,  fork_real_y,  c="#1565C0", s=55, alpha=0.9, marker="^")
    ax.scatter(knife_syn_x,  knife_syn_y,  c="#FFCC80", s=30, alpha=0.6, marker="s")
    ax.scatter(knife_real_x, knife_real_y, c="#BF360C", s=55, alpha=0.9, marker="s")
    ax.plot(*PLATE_POS, "o", ms=12, color="#555", zorder=5)

    # manual legend below the plot
    legend_elements = [
        plt.Line2D([0],[0], color="#1565C0", lw=1.5, linestyle="--", label="Franka workspace boundary"),
        plt.scatter([],[],  c="#1565C0", s=55, marker="^",           label=f"Fork — real UMI (n={len(fork_real_x)})"),
        plt.scatter([],[],  c="#90CAF9", s=30, marker="^",           label=f"Fork — synthetic (n={len(fork_syn_x)})"),
        plt.scatter([],[],  c="#BF360C", s=55, marker="s",           label=f"Knife — real UMI (n={len(knife_real_x)})"),
        plt.scatter([],[],  c="#FFCC80", s=30, marker="s",           label=f"Knife — synthetic (n={len(knife_syn_x)})"),
        plt.Rectangle((0,0),1,1, fc="#2196F3", alpha=0.3,            label="Eval range — fork"),
        plt.Rectangle((0,0),1,1, fc="#FF5722", alpha=0.3,            label="Eval range — knife"),
        plt.Line2D([0],[0], marker="o", ms=9, color="#555", lw=0,    label="Plate (fixed)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower left",
              bbox_to_anchor=(0.0, -0.38), ncol=3, frameon=True)

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title("Object Starting Positions: Training Data vs Eval Range", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.10, 0.90)
    ax.set_ylim(-0.80, 0.15)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    plt.subplots_adjust(bottom=0.30)
    out = OUT_DIR / "fig2_spatial_coverage.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 3: UMI filtering ──────────────────────────────────────────────────
def fig_umi_filtering():
    with open(DATA_DIR / "object_poses.json") as f:
        raw = json.load(f)

    valid_f, valid_k, invalid_f, invalid_k = [], [], [], []

    for ep in raw:
        if ep.get("status") != "full":
            continue
        pos = {o["object_name"]: to_world(o["tvec"])
               for o in ep["objects"] if o["object_name"] != "plate"}
        ok = all(in_workspace(*p) for p in pos.values())
        for name, (wx, wy) in pos.items():
            bucket_v = valid_f   if name == "fork"  else valid_k
            bucket_i = invalid_f if name == "fork"  else invalid_k
            (bucket_v if ok else bucket_i).append((wx, wy))

    fig, ax = plt.subplots(figsize=(7, 7))

    # workspace (annulus)
    ox, oy = workspace_arc(*ROBOT_BASE, R_MAX_H)
    ix, iy = workspace_arc(*ROBOT_BASE, R_MIN_H)
    ax.fill(ox, oy, color="#e8f5e9", alpha=0.4, zorder=0)
    ax.plot(ox, oy, color="#2e7d32", lw=2, linestyle="--", label="Franka workspace boundary")
    ax.fill(ix, iy, color="white", alpha=1.0, zorder=1)

    if invalid_f:
        ax.scatter(*zip(*invalid_f), c="#ef9a9a", s=50, marker="^", alpha=0.75, zorder=3)
    if invalid_k:
        ax.scatter(*zip(*invalid_k), c="#ef9a9a", s=50, marker="s", alpha=0.75, zorder=3)
    if valid_f:
        ax.scatter(*zip(*valid_f),   c="#1b5e20", s=65, marker="^", alpha=0.9,  zorder=4)
    if valid_k:
        ax.scatter(*zip(*valid_k),   c="#1b5e20", s=65, marker="s", alpha=0.9,  zorder=4)

    ax.plot(*PLATE_POS, "o", ms=12, color="#555", zorder=5)
    ax.plot(*ROBOT_BASE, "kD", ms=10, zorder=6)

    nv = len(set(map(id, valid_f)))   # approx per-episode count
    legend_elements = [
        plt.Line2D([0],[0], color="#2e7d32", lw=2, linestyle="--", label="Franka workspace boundary"),
        plt.scatter([],[],  c="#1b5e20", s=65, marker="^", label=f"Fork — valid ({len(valid_f)})"),
        plt.scatter([],[],  c="#1b5e20", s=65, marker="s", label=f"Knife — valid ({len(valid_k)})"),
        plt.scatter([],[],  c="#ef9a9a", s=50, marker="^", label=f"Fork — discarded ({len(invalid_f)})"),
        plt.scatter([],[],  c="#ef9a9a", s=50, marker="s", label=f"Knife — discarded ({len(invalid_k)})"),
        plt.Line2D([0],[0], marker="o",  ms=9,  color="#555", lw=0, label="Plate (fixed)"),
        plt.Line2D([0],[0], marker="D",  ms=9,  color="k",    lw=0, label="Robot base (0.35, −0.74)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower left",
              bbox_to_anchor=(0.0, -0.30), ncol=2, frameon=True)

    n_valid_ep  = sum(1 for e in raw if e.get("status") == "full"
                      and all(in_workspace(*to_world(o["tvec"]))
                              for o in e["objects"] if o["object_name"] != "plate"))
    n_total_ep  = sum(1 for e in raw if e.get("status") == "full")

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(f"UMI Data Filtering: {n_valid_ep} valid / {n_total_ep} episodes  "
                 f"({n_valid_ep/n_total_ep*100:.0f}% usable)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-0.30, 1.00)
    ax.set_ylim(-0.85, 0.35)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    plt.subplots_adjust(bottom=0.25)
    out = OUT_DIR / "fig3_umi_filtering.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── summary ───────────────────────────────────────────────────────────────────
def print_summary():
    with open(DATA_DIR / "object_poses.json") as f:
        raw = json.load(f)
    total = sum(1 for e in raw if e.get("status") == "full")
    valid = sum(1 for e in raw if e.get("status") == "full" and
                all(in_workspace(*to_world(o["tvec"]))
                    for o in e["objects"] if o["object_name"] != "plate"))
    with open(DATA_DIR / "object_poses_combined.json") as f:
        combined = json.load(f)
    real_n = sum(1 for e in combined if not e["video_name"].startswith("synthetic"))
    syn_n  = sum(1 for e in combined if e["video_name"].startswith("synthetic"))

    print("=" * 52)
    print("DATA QUALITY SUMMARY")
    print("=" * 52)
    print(f"UMI episodes (full):           {total}")
    print(f"  in workspace (annular+table): {valid}  ({valid/total*100:.1f}%)")
    print(f"  discarded:                    {total-valid}  ({(total-valid)/total*100:.1f}%)")
    print(f"Synthetic episodes:             {syn_n}")
    print(f"Combined base:                  {real_n + syn_n}")
    print(f"Datagen target (×10):           {(real_n + syn_n) * 10}")
    print("=" * 52)


if __name__ == "__main__":
    print_summary()
    fig_pipeline()
    fig_spatial_coverage()
    fig_umi_filtering()
    print(f"\nAll figures saved to: {OUT_DIR}")
