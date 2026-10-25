import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects

ROOT = Path.cwd()
BASE = ROOT / "artifacts/common/lda/lda_k15_seed42"
OUT = ROOT / "results/lda/figures"
OUT.mkdir(parents=True, exist_ok=True)

# Load original artifacts
topics = pd.read_csv(BASE / "topic_summary.csv", encoding="utf-8-sig")
projection = pd.read_csv(BASE / "projection_2d.csv", encoding="utf-8-sig")

with open(BASE / "topic_display_mapping.json", encoding="utf-8") as f:
    mapping = json.load(f)

labels = {
    int(k): v["display_name"]
    for k, v in mapping["mapping"].items()
}

assert len(labels) == 15, "Expected 15 topic labels"
assert topics["document_count"].sum() == 9274
assert set(topics["topic_id"]) == set(labels)

# Use font supporting Vietnamese characters
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["savefig.dpi"] = 220

# ==========================================
# F05 - Topic distribution with proper names
# ==========================================

df = topics.sort_values("document_count").copy()

fig, ax = plt.subplots(figsize=(15, 9))

colors = [
    "#3269A8" if int(t) % 2 == 0 else "#07988B"
    for t in df["topic_id"]
]

bars = ax.barh(
    range(len(df)),
    df["document_count"],
    color=colors
)

ax.set_yticks(range(len(df)))
ax.set_yticklabels([
    f"T{int(t):02d} - {labels[int(t)]}"
    for t in df["topic_id"]
])

ax.bar_label(bars, padding=4)

ax.set_xlim(0, df["document_count"].max() * 1.18)
ax.set_xlabel("Number of articles")
ax.set_title(
    "LDA Final (K=15) - Topic Distribution",
    fontsize=15,
    weight="bold"
)

ax.grid(axis="x", linestyle="--", alpha=0.2)
ax.set_axisbelow(True)

fig.text(
    0.02, 0.015,
    "Topic names are proposed labels; independent review pending.",
    fontsize=9, color="#58677B"
)

fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(
    OUT / "f05_final_topic_distribution.png",
    bbox_inches="tight"
)
plt.close(fig)

print("DONE F05")

# ==========================================
# F07 - PCA plot with Vietnamese topic names
# ==========================================

fig, ax = plt.subplots(figsize=(17, 10))
palette = plt.get_cmap("tab20")

for i, tid in enumerate(sorted(labels)):
    points = projection[
        projection["topic_id"] == tid
    ]

    color = palette(i % 20)

    ax.scatter(
        points["x"],
        points["y"],
        s=7,
        alpha=0.45,
        color=color,
        label=f"T{tid:02d} - {labels[tid]}",
        rasterized=True
    )

    if not points.empty:
        cx = points["x"].median()
        cy = points["y"].median()

        text = ax.text(
            cx, cy, f"T{tid:02d}",
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="#888888",
                boxstyle="round,pad=0.25",
                alpha=0.85
            ),
            zorder=10
        )

ax.set_title(
    "LDA Final (K=15) - PCA Topic Projection",
    fontsize=15,
    weight="bold"
)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")

ax.legend(
    title="Proposed Topic Names",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=9,
    markerscale=3
)

ax.grid(alpha=0.15)

fig.text(
    0.02, 0.015,
    "PCA visualization of document-topic probabilities; proposed labels.",
    fontsize=9, color="#58677B"
)

fig.tight_layout(rect=[0, 0.03, 0.99, 1])
fig.savefig(
    OUT / "f07_final_projection_2d.png",
    bbox_inches="tight"
)
plt.close(fig)

print("DONE F07")
print("Saved to:", OUT)
