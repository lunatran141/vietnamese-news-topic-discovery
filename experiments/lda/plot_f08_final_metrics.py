import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path.cwd()
METRICS = ROOT / "results/lda/final/test_metrics.json"
OUT = ROOT / "results/lda/figures"
OUT.mkdir(parents=True, exist_ok=True)

# Đọc chỉ số từ kết quả thực nghiệm, không nhập số giả
with METRICS.open("r", encoding="utf-8") as f:
    data = json.load(f)

print("Available metric fields:", list(data.keys()))

# Tìm giá trị trong JSON, kể cả trường hợp nested
def find_metric(obj, names):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in names and isinstance(value, (int, float)):
                return float(value)
        for value in obj.values():
            result = find_metric(value, names)
            if result is not None:
                return result
    return None

metrics = [
    ("NPMI Coherence", ["npmi_coherence", "npmi", "coherence_npmi", "npmi_common"],
     "Topic keyword coherence"),
    ("Topic Diversity", ["topic_diversity", "diversity"],
     "Keyword diversity across topics"),
    ("ARI", ["ari", "adjusted_rand_index"],
     "Agreement with category_gold"),
    ("NMI", ["nmi", "normalized_mutual_info"],
     "Shared information with category_gold"),
]

values = []

for title, keys, description in metrics:
    value = find_metric(data, keys)
    if value is None:
        raise ValueError(
            f"Missing {title} in {METRICS}. "
            f"Available keys: {list(data.keys())}"
        )
    values.append((title, value, description))

# Bố cục 16:9 dành cho PowerPoint
fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
fig.patch.set_facecolor("#F8FAFC")
ax.set_facecolor("#F8FAFC")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

ax.text(
    0.9, 8.15,
    "LDA FINAL TEST RESULTS",
    fontsize=27, fontweight="bold",
    color="#152B45"
)

ax.text(
    0.9, 7.55,
    "K = 15  |  Seed = 42  |  Independent Test = 1,390 articles",
    fontsize=15, color="#52657C"
)

colors = ["#2563EB", "#0891B2", "#7C3AED", "#0F766E"]

positions = [
    (0.9, 4.15),
    (8.2, 4.15),
    (0.9, 1.65),
    (8.2, 1.65)
]

for i, ((title, value, description), (x, y)) in enumerate(
    zip(values, positions)
):
    box = FancyBboxPatch(
        (x, y), 6.9, 2.15,
        boxstyle="round,pad=0.08,rounding_size=0.22",
        linewidth=1.2,
        edgecolor="#DFE7F0",
        facecolor="white"
    )
    ax.add_patch(box)

    ax.text(
        x + 0.35, y + 1.58,
        title,
        fontsize=17, fontweight="bold",
        color="#334155"
    )

    ax.text(
        x + 0.35, y + 0.78,
        f"{value:.4f}",
        fontsize=32, fontweight="bold",
        color=colors[i]
    )

    ax.text(
        x + 0.35, y + 0.30,
        description,
        fontsize=12, color="#64748B"
    )

ax.text(
    0.9, 0.65,
    "Note: ARI and NMI measure agreement with editorial categories, "
    "not topic-label accuracy.",
    fontsize=12, color="#475569"
)

ax.text(
    0.9, 0.26,
    "Source: results/lda/final/test_metrics.json",
    fontsize=10, color="#94A3B8"
)

plt.subplots_adjust(0, 0, 1, 1)

output = OUT / "f08_final_test_metrics.png"

plt.savefig(
    output,
    dpi=220,
    facecolor=fig.get_facecolor()
)

plt.close(fig)

print("F08 CREATED:", output)
print("Values:", [(name, round(v, 6)) for name, v, _ in values])

