
"""Create 9 LDA research figures from frozen project artifacts.

Run from repository root:
    python experiments/lda/plot_lda_results.py

No model fitting or changes to source artifacts.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.article_text import build_article_text
from src.models.lda_pipeline import transform_lda_pipeline


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RESULTS = ROOT / "results" / "lda"

COMMON = (
    ROOT
    / "artifacts"
    / "common"
    / "lda"
    / "lda_k15_seed42"
)

FINAL_MODEL = (
    ROOT
    / "artifacts"
    / "lda"
    / "final"
    / "lda_pipeline.joblib"
)

FIGURES = RESULTS / "figures"

FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. VISUAL STYLE
# ============================================================

COLORS = {
    "blue": "#3269A8",
    "teal": "#07988B",
    "gold": "#E4A33A",
    "orange": "#DC7649",
    "purple": "#7956B3",
    "red": "#C94D55",
    "muted": "#58677B",
    "light": "#ECF1F5",
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
    }
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def read_csv(
    path: Path,
    required: set[str],
) -> pd.DataFrame:
    """Read CSV and validate required columns."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing input: {path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} missing columns: "
            f"{sorted(missing)}"
        )

    return df


def format_axis(ax, axis="y"):
    """Apply a consistent style to axes."""

    ax.grid(
        axis=axis,
        linestyle="--",
        alpha=0.22,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    """Save and close a figure."""

    path = FIGURES / name

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("Saved:", path)


def short_label(raw, max_len=26):
    """Shorten topic labels for plotting."""

    text = (
        str(raw)
        .replace("_", " ")
        .replace(" · ", " ")
        .strip()
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    if len(text) <= max_len:
        return text

    return text[:max_len - 1] + "…"


# ============================================================
# 4. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD EXPERIMENT DATA
    # --------------------------------------------------------

    m08 = read_csv(
        RESULTS / "m08/topic_number_results.csv",
        {
            "k",
            "coherence_cv",
            "coherence_npmi",
            "topic_diversity",
            "runtime_seconds",
        },
    ).sort_values("k")

    m09 = read_csv(
        RESULTS / "m09/pairwise_stability.csv",
        {
            "seed_a",
            "seed_b",
            "k",
            "mean_matched_jaccard",
            "mean_document_cosine",
        },
    )

    m09 = (
        m09.loc[m09["k"] == 15]
        .sort_values(["seed_a", "seed_b"])
        .copy()
    )

    topics = read_csv(
        COMMON / "topic_summary.csv",
        {
            "topic_id",
            "topic_name",
            "document_count",
            "percentage",
        },
    ).sort_values("topic_id")

    keywords = read_csv(
        COMMON / "topic_keywords.csv",
        {
            "topic_id",
            "rank",
            "keyword",
            "weight",
        },
    )

    npmi = read_csv(
        RESULTS / "final/topic_keywords.csv",
        {
            "topic_id",
            "npmi_common",
        },
    )

    # Validate final topic count
    if (
        len(topics) != 15
        or topics["topic_id"].nunique() != 15
    ):
        raise ValueError(
            "Expected 15 final LDA topics"
        )

    # Validate total number of documents
    if int(topics["document_count"].sum()) != 9274:
        raise ValueError(
            "Expected 9274 document assignments "
            "across all splits"
        )

    print("\n======================================")
    print("BUILDING LDA FIGURES")
    print("Model: LDA Final K=15 Seed=42")
    print("======================================\n")

    # ========================================================
    # F01 - M08 METRICS COMPARISON
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.5),
        sharex=True,
    )

    metrics = [
        (
            "coherence_cv",
            "Coherence C_V",
            COLORS["blue"],
        ),
        (
            "coherence_npmi",
            "Coherence C_NPMI",
            COLORS["orange"],
        ),
        (
            "topic_diversity",
            "Topic Diversity",
            COLORS["teal"],
        ),
    ]

    for ax, (column, label, color) in zip(
        axes,
        metrics,
    ):

        ax.plot(
            m08["k"],
            m08[column],
            marker="o",
            linewidth=2.2,
            color=color,
            label=label,
        )

        ax.axvline(
            15,
            color=COLORS["red"],
            linestyle="--",
            linewidth=1.5,
            label="Selected K=15",
        )

        selected_value = float(
            m08.loc[
                m08["k"] == 15,
                column,
            ].iloc[0]
        )

        ax.scatter(
            [15],
            [selected_value],
            color=COLORS["red"],
            zorder=5,
        )

        ax.set_title(label)
        ax.set_xlabel("Number of Topics (K)")
        ax.set_xticks(m08["k"])

        ax.legend(fontsize=8)

        format_axis(ax)

    fig.suptitle(
        "M08 | LDA Topic-Number Selection on Validation",
        weight="bold",
        fontsize=15,
    )

    fig.tight_layout()

    save(
        fig,
        "f01_m08_metrics_comparison.png",
    )

    # ========================================================
    # F02 - M08 RUNTIME VS K
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10.5, 5.5)
    )

    ax.plot(
        m08["k"],
        m08["runtime_seconds"],
        marker="o",
        linewidth=2.4,
        color=COLORS["purple"],
        label="Measured Runtime",
    )

    ax.axvline(
        15,
        color=COLORS["red"],
        linestyle="--",
        label="Selected K=15",
    )

    for row in m08.itertuples():

        ax.annotate(
            f"{row.runtime_seconds:.1f}s",
            (
                row.k,
                row.runtime_seconds,
            ),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_title(
        "M08 | LDA Training Runtime by Number of Topics"
    )

    ax.set_xlabel("Number of Topics (K)")
    ax.set_ylabel("Runtime (Seconds)")

    ax.set_xticks(m08["k"])

    ax.legend()

    format_axis(ax)

    fig.tight_layout()

    save(
        fig,
        "f02_m08_runtime_vs_k.png",
    )

    # ========================================================
    # F03 - M09 PAIRWISE STABILITY
    # ========================================================

    seed_labels = [
        f"{int(a)}–{int(b)}"
        for a, b in zip(
            m09["seed_a"],
            m09["seed_b"],
        )
    ]

    x = np.arange(len(m09))

    width = 0.38

    fig, ax = plt.subplots(
        figsize=(13, 5.7)
    )

    ax.bar(
        x - width / 2,
        m09["mean_matched_jaccard"],
        width,
        color=COLORS["blue"],
        label="Matched Keyword Jaccard",
    )

    ax.bar(
        x + width / 2,
        m09["mean_document_cosine"],
        width,
        color=COLORS["gold"],
        label="Document-Topic Cosine",
    )

    ax.set_xticks(
        x,
        seed_labels,
        rotation=30,
        ha="right",
    )

    ax.set_ylim(0, 1.04)

    ax.set_title(
        "M09 | Pairwise Stability of LDA (K=15)"
    )

    ax.set_xlabel("Seed Pair")
    ax.set_ylabel("Similarity")

    ax.legend(loc="upper right")

    format_axis(ax)

    fig.tight_layout()

    save(
        fig,
        "f03_m09_stability_pairs.png",
    )

    # ========================================================
    # F04 - M09 RUNTIME BY SEED
    # ========================================================

    runtime = read_csv(
        RESULTS / "m09/runtime_by_seed.csv",
        set(),
    )

    seed_candidates = [
        column
        for column in (
            "seed",
            "random_state",
        )
        if column in runtime.columns
    ]

    time_candidates = [
        column
        for column in (
            "runtime_seconds",
            "fit_seconds",
            "total_runtime_seconds",
        )
        if column in runtime.columns
    ]

    if not seed_candidates or not time_candidates:

        print(
            "SKIPPED F04: Missing compatible "
            "runtime columns:",
            list(runtime.columns),
        )

    else:

        seed_column = seed_candidates[0]
        time_column = time_candidates[0]

        points = (
            runtime[
                [seed_column, time_column]
            ]
            .dropna()
            .groupby(
                seed_column,
                as_index=False,
            )[time_column]
            .mean()
        )

        fig, ax = plt.subplots(
            figsize=(9.5, 5)
        )

        bars = ax.bar(
            points[seed_column].astype(str),
            points[time_column],
            color=COLORS["teal"],
        )

        ax.bar_label(
            bars,
            fmt="%.1f s",
            padding=4,
        )

        ax.set_title(
            "M09 | Measured Runtime by Random Seed"
        )

        ax.set_xlabel("Random Seed")
        ax.set_ylabel("Runtime (Seconds)")

        ax.set_ylim(
            0,
            points[time_column].max() * 1.18,
        )

        format_axis(ax)

        fig.tight_layout()

        save(
            fig,
            "f04_m09_runtime_by_seed.png",
        )

    # ========================================================
    # F05 - FINAL TOPIC DISTRIBUTION
    # ========================================================

    ordered = topics.sort_values(
        "document_count"
    )

    topic_labels = [
        (
            f"T{int(row.topic_id):02d}  "
            f"{short_label(row.topic_name, 28)}"
        )
        for row in ordered.itertuples()
    ]

    fig, ax = plt.subplots(
        figsize=(12.5, 8)
    )

    colors = [
        (
            COLORS["blue"]
            if int(topic_id) % 2 == 0
            else COLORS["teal"]
        )
        for topic_id in ordered["topic_id"]
    ]

    bars = ax.barh(
        topic_labels,
        ordered["document_count"],
        color=colors,
    )

    ax.bar_label(
        bars,
        padding=4,
        fontsize=9,
    )

    ax.set_xlim(
        0,
        ordered["document_count"].max() * 1.18,
    )

    ax.set_title(
        "LDA Final | Topic Sizes Across "
        "Train + Validation + Test"
    )

    ax.set_xlabel("Number of Articles")

    format_axis(
        ax,
        "x",
    )

    fig.tight_layout()

    save(
        fig,
        "f05_final_topic_distribution.png",
    )

    # ========================================================
    # F06 - FINAL TOPIC NPMI
    # ========================================================

    metric = (
        npmi.groupby(
            "topic_id",
            as_index=False,
        )["npmi_common"]
        .first()
        .sort_values("topic_id")
    )

    fig, ax = plt.subplots(
        figsize=(11, 5.6)
    )

    values = metric[
        "npmi_common"
    ].astype(float)

    bars = ax.bar(
        [
            f"T{int(topic_id)}"
            for topic_id in metric["topic_id"]
        ],
        values,
        color=[
            (
                COLORS["teal"]
                if value >= 0
                else COLORS["red"]
            )
            for value in values
        ],
    )

    ax.axhline(
        values.mean(),
        color=COLORS["purple"],
        linestyle="--",
        label=(
            "Mean of Topic NPMI = "
            f"{values.mean():.3f}"
        ),
    )

    ax.axhline(
        0,
        color=COLORS["muted"],
        linewidth=0.7,
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.annotate(
            f"{value:.3f}",
            (
                bar.get_x()
                + bar.get_width() / 2,
                value,
            ),
            xytext=(
                0,
                5 if value >= 0 else -13,
            ),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_title(
        "LDA Final | Topic-Level NPMI"
    )

    ax.set_xlabel("Topic")
    ax.set_ylabel("NPMI")

    ax.legend()

    format_axis(ax)

    fig.tight_layout()

    save(
        fig,
        "f06_final_topic_npmi.png",
    )

    # ========================================================
    # F07 - FINAL 2D PCA PROJECTION
    # ========================================================

    projection = read_csv(
        COMMON / "projection_2d.csv",
        {
            "x",
            "y",
            "topic_id",
            "split",
        },
    )

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    palette = plt.get_cmap("tab20")

    topic_ids = sorted(
        projection[
            "topic_id"
        ].dropna().astype(int).unique()
    )

    for index, topic_id in enumerate(
        topic_ids
    ):

        selected = projection.loc[
            projection["topic_id"]
            == topic_id
        ]

        ax.scatter(
            selected["x"],
            selected["y"],
            s=5,
            alpha=0.45,
            color=palette(index % 20),
            label=f"T{topic_id}",
            rasterized=True,
        )

    ax.set_title(
        "LDA Final | PCA Projection of "
        "Document-Topic Probabilities"
    )

    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    ax.legend(
        title="Topic",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        ncol=1,
    )

    format_axis(
        ax,
        "both",
    )

    fig.tight_layout()

    save(
        fig,
        "f07_final_projection_2d.png",
    )

    # ========================================================

    print(
        "\nFinished creating "
        "available LDA figures."
    )


if __name__ == "__main__":
    main()
