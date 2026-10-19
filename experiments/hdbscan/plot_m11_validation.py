"""Vẽ kết quả HDBSCAN trên validation từ CSV đã lưu."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/hdbscan/validation/m11_validation_metrics.csv"
OUTPUT = ROOT / "results/hdbscan/figures"

RUNS = [
    ("minilm_title_content", "eom_15_5", "MiniLM\nEOM 15/5"),
    ("minilm_title_content", "leaf_15_5", "MiniLM\nLeaf 15/5"),
    ("mpnet_title_content", "eom_15_5", "MPNet\nEOM 15/5"),
    ("mpnet_title_content", "leaf_15_5", "MPNet\nLeaf 15/5"),
]


def load_results() -> pd.DataFrame:
    data = pd.read_csv(INPUT)
    required = {
        "embedding_run", "experiment", "split", "n_articles",
        "n_noise", "noise_percent", "coverage_percent", "ari", "nmi",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {sorted(missing)}")

    rows = []
    for embedding, experiment, label in RUNS:
        match = data.loc[
            (data["embedding_run"] == embedding)
            & (data["experiment"] == experiment)
            & (data["split"] == "val")
        ]
        if len(match) != 1:
            raise ValueError(
                f"Cần đúng một dòng cho {embedding} + {experiment}; "
                f"hiện có {len(match)}"
            )

        row = match.iloc[0].copy()
        row["display_label"] = label
        rows.append(row)

    result = pd.DataFrame(rows).reset_index(drop=True)
    numeric = [
        "n_articles", "n_noise", "noise_percent",
        "coverage_percent", "ari", "nmi",
    ]
    result[numeric] = result[numeric].apply(
        pd.to_numeric, errors="raise"
    )

    if not (result["n_articles"] == 1391).all():
        raise ValueError("Các run phải dùng cùng 1.391 bài validation")

    if not np.allclose(
        result["noise_percent"] + result["coverage_percent"],
        100,
        atol=0.02,
    ):
        raise ValueError("Tỷ lệ nhiễu và độ bao phủ không cộng thành 100%")

    calculated_noise = (
        result["n_noise"] / result["n_articles"] * 100
    )
    if not np.allclose(
        calculated_noise, result["noise_percent"], atol=0.01
    ):
        raise ValueError("n_noise không khớp noise_percent")

    return result


def plot_noise_coverage(data: pd.DataFrame) -> Path:
    x = np.arange(len(data))
    coverage = data["coverage_percent"].to_numpy()
    noise = data["noise_percent"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    assigned_bars = ax.bar(
        x, coverage, width=0.6, color="#4C78A8",
        label="Assigned to a cluster",
    )
    noise_bars = ax.bar(
        x, noise, bottom=coverage, width=0.6,
        color="#B8B8B8", label="Outlier (-1)",
    )

    ax.bar_label(
        assigned_bars, labels=[f"{v:.2f}%" for v in coverage],
        label_type="center", color="white", fontsize=10,
    )
    ax.bar_label(
        noise_bars, labels=[f"{v:.2f}%" for v in noise],
        label_type="center", fontsize=10,
    )

    ax.set_xticks(x, data["display_label"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage of validation articles")
    ax.set_title("HDBSCAN validation: assigned articles and outliers")
    ax.legend(loc="upper right")

    path = OUTPUT / "m11_validation_noise_coverage.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ari_nmi(data: pd.DataFrame) -> Path:
    x = np.arange(len(data))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ari_bars = ax.bar(
        x - width / 2, data["ari"], width,
        color="#7A5195", label="ARI",
    )
    nmi_bars = ax.bar(
        x + width / 2, data["nmi"], width,
        color="#2A9D8F", label="NMI",
    )

    ax.bar_label(ari_bars, fmt="%.4f", padding=3, fontsize=9)
    ax.bar_label(nmi_bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xticks(x, data["display_label"])
    ax.set_ylim(0, 0.52)
    ax.set_ylabel("Score")
    ax.set_title("HDBSCAN validation: ARI and NMI")
    ax.legend()

    path = OUTPUT / "m11_validation_ari_nmi.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    data = load_results()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for path in (
        plot_noise_coverage(data),
        plot_ari_nmi(data),
    ):
        print(f"Đã lưu: {path}")


if __name__ == "__main__":
    main()