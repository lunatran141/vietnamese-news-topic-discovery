"""Vẽ ba biểu đồ M11 từ các CSV đã lưu."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]

LABELS = {
    "minilm_title_content": "MiniLM: tiêu đề + 60 từ",
    "mpnet_title_content": "MPNet: tiêu đề + 60 từ",
    "minilm_title_only": "MiniLM: chỉ tiêu đề",
    "mpnet_title_only": "MPNet: chỉ tiêu đề",
}

COLORS = {
    "minilm_title_content": "#92B5D8",
    "mpnet_title_content": "#6FAA79",
    "minilm_title_only": "#DB8892",
    "mpnet_title_only": "#D3DA9D",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
})


def plot_comparison(
    summary,
    embedding_names,
    experiment_names,
    metric,
    title,
    ylabel,
    output_path,
    percentage=False,
):
    """Vẽ các cột so sánh bốn embedding ở tám cấu hình."""
    data = summary.pivot(
        index="experiment",
        columns="embedding_run",
        values=metric,
    ).reindex(
        index=experiment_names,
        columns=embedding_names,
    )

    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Thiếu hoặc sai số liệu cho biểu đồ {metric}")

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    x = np.arange(len(experiment_names))
    width = 0.19

    for index, embedding_name in enumerate(embedding_names):
        heights = data[embedding_name].to_numpy(dtype=float)
        positions = x + (index - (len(embedding_names) - 1) / 2) * width
        bars = ax.bar(
            positions,
            heights,
            width=width,
            label=LABELS[embedding_name],
            color=COLORS[embedding_name],
        )

        labels = [
            f"{value:.1f}" if percentage else f"{int(value)}"
            for value in heights
        ]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names, rotation=25, ha="right")
    ax.set_xlabel("Cấu hình HDBSCAN")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=55)
    ax.set_ylim(0, max(1.0, float(values.max()) * 1.2))

    if percentage:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu: {output_path}")


def plot_cluster_sizes(assignments, output_path):
    """So sánh kích thước cụm EOM và Leaf của MPNet title + content."""
    experiments = ["eom_15_5", "leaf_15_5"]
    distributions = {}

    for experiment in experiments:
        group = assignments.loc[
            assignments["experiment"] == experiment
        ]
        if group.empty:
            raise ValueError(f"Không có kết quả {experiment}")

        # Nhãn -1 là nhiễu, không tính như một cụm.
        sizes = group.loc[
            group["cluster_id"] != -1, "cluster_id"
        ].value_counts()

        if sizes.empty:
            raise ValueError(f"{experiment} không có cụm để vẽ")

        distributions[experiment] = sizes

    largest = max(
        int(sizes.max()) for sizes in distributions.values()
    )

    # Dùng cùng khoảng chia để hai biểu đồ có thể so sánh.
    bin_width = 20
    upper = int(np.ceil(largest / bin_width) * bin_width)
    bins = np.arange(0, upper + bin_width, bin_width)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, experiment, color in zip(
        axes, experiments, ["#4477AA", "#228833"]
    ):
        sizes = distributions[experiment]

        ax.hist(
            sizes.to_numpy(),
            bins=bins,
            color=color,
            edgecolor="white",
        )
        ax.set_title(
            f"{experiment}\n"
            f"{len(sizes)} cụm; lớn nhất {int(sizes.max())} bài"
        )
        ax.set_xlabel("Số bài trong một cụm")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Số cụm")
    fig.suptitle(
        "MPNet: tiêu đề + 60 từ — phân bố kích thước cụm",
        fontsize=15,
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu: {output_path}")


def main():
    with (ROOT / "configs/hdbscan.yaml").open(
        encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    output_dir = ROOT / config["output_dir"]
    figures_dir = output_dir / "figures"

    summary = pd.read_csv(
        output_dir / "m11_summary.csv",
        encoding="utf-8-sig",
    )
    embedding_names = list(config["embedding_artifacts"])
    experiment_names = [
        experiment["name"]
        for experiment in config["hdbscan"]["experiments"]
    ]

    expected = {
        (embedding, experiment)
        for embedding in embedding_names
        for experiment in experiment_names
    }
    actual = set(
        summary[["embedding_run", "experiment"]]
        .itertuples(index=False, name=None)
    )
    if actual != expected or len(summary) != len(expected):
        raise ValueError("Summary thiếu cấu hình hoặc có dòng bị trùng")

    assignments = pd.read_csv(
        output_dir / "mpnet_title_content_assignments.csv",
        encoding="utf-8-sig",
    )

    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_comparison(
        summary,
        embedding_names,
        experiment_names,
        metric="noise_percent",
        title="M11: tỷ lệ bài nhiễu trên tập train",
        ylabel="Tỷ lệ bài nhiễu",
        output_path=figures_dir / "m11_noise_ratio.png",
        percentage=True,
    )

    plot_comparison(
        summary,
        embedding_names,
        experiment_names,
        metric="n_clusters",
        title="M11: số cụm theo cấu hình HDBSCAN",
        ylabel="Số cụm, không tính nhiễu",
        output_path=figures_dir / "m11_cluster_count.png",
    )

    plot_cluster_sizes(
        assignments,
        figures_dir / "m11_cluster_size_distribution.png",
    )

if __name__ == "__main__":
    main()