"""Tạo các biểu đồ M11 từ CSV và embedding artifact đã lưu."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from umap import UMAP
import yaml

from src.features.embeddings import load_embedding_artifact


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

SIZE_FIGURE_NAMES = {
    "minilm_title_content": "m11_minilm_cluster_size_distribution.png",
    "mpnet_title_content": "m11_cluster_size_distribution.png",
    "minilm_title_only": (
        "m11_minilm_title_only_cluster_size_distribution.png"
    ),
    "mpnet_title_only": (
        "m11_mpnet_title_only_cluster_size_distribution.png"
    ),
}

UMAP_2D_EMBEDDINGS = (
    "minilm_title_content",
    "mpnet_title_content",
)

UMAP_2D_EXPERIMENTS = (
    "eom_15_5",
    "leaf_15_5",
)

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
    """Vẽ cột so sánh bốn embedding trên tám cấu hình HDBSCAN."""
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

    fig, ax = plt.subplots(
        figsize=(14, 7),
        constrained_layout=True,
    )
    x = np.arange(len(experiment_names))
    width = 0.19

    for index, embedding_name in enumerate(embedding_names):
        heights = data[embedding_name].to_numpy(dtype=float)
        positions = (
            x
            + (index - (len(embedding_names) - 1) / 2) * width
        )

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
        ax.bar_label(
            bars,
            labels=labels,
            padding=3,
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        experiment_names,
        rotation=25,
        ha="right",
    )
    ax.set_xlabel("Cấu hình HDBSCAN")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=55)
    ax.set_ylim(
        0,
        max(1.0, float(values.max()) * 1.2),
    )

    if percentage:
        ax.yaxis.set_major_formatter(
            PercentFormatter(xmax=100)
        )
    else:
        ax.yaxis.set_major_locator(
            MaxNLocator(integer=True)
        )

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Đã lưu: {output_path}")


def plot_cluster_sizes(
    assignments,
    output_path,
    embedding_name,
):
    """So sánh kích thước cụm EOM và Leaf của một embedding."""
    experiments = ("eom_15_5", "leaf_15_5")
    distributions = {}

    for experiment in experiments:
        group = assignments.loc[
            assignments["experiment"] == experiment
        ]

        if group.empty:
            raise ValueError(
                f"{embedding_name}: không có kết quả {experiment}"
            )

        # Nhãn -1 là nhiễu, không tính như một cụm.
        sizes = group.loc[
            group["cluster_id"] != -1,
            "cluster_id",
        ].value_counts()

        if sizes.empty:
            raise ValueError(
                f"{embedding_name}, {experiment}: không có cụm"
            )

        distributions[experiment] = sizes

    largest = max(
        int(sizes.max())
        for sizes in distributions.values()
    )

    # Cùng độ rộng khoảng chia cho EOM và Leaf.
    bin_width = 20
    upper = int(
        np.ceil(largest / bin_width) * bin_width
    )
    bins = np.arange(
        0,
        upper + bin_width,
        bin_width,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, experiment, color in zip(
        axes,
        experiments,
        ("#4477AA", "#228833"),
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
            f"{len(sizes)} cụm; "
            f"lớn nhất {int(sizes.max())} bài"
        )
        ax.set_xlabel("Số bài trong một cụm")
        ax.yaxis.set_major_locator(
            MaxNLocator(integer=True)
        )
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Số cụm")
    fig.suptitle(
        f"{LABELS[embedding_name]} "
        "— phân bố kích thước cụm",
        fontsize=15,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Đã lưu: {output_path}")


def plot_umap_2d(
    embedding_name,
    embedding_table,
    assignments,
    umap_config,
    figures_dir,
):
    """
    Chiếu embedding xuống 2D để minh họa nhãn M11 đã lưu.

    Nhãn cụm được đọc từ assignments; hàm này không chạy HDBSCAN.
    """
    article_ids = embedding_table["article_id"].tolist()

    if len(article_ids) != len(set(article_ids)):
        raise ValueError(
            f"{embedding_name}: article_id trùng trong artifact"
        )

    vectors = np.stack(
        embedding_table["embedding"].to_numpy()
    ).astype(np.float32)

    if (
        vectors.ndim != 2
        or len(vectors) != len(article_ids)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError(
            f"{embedding_name}: ma trận embedding không hợp lệ"
        )

    params_2d = dict(umap_config)
    params_2d["n_components"] = 2

    print(f"Đang tạo UMAP 2D: {embedding_name}")
    points = UMAP(**params_2d).fit_transform(vectors)

    if (
        points.shape != (len(article_ids), 2)
        or not np.isfinite(points).all()
    ):
        raise ValueError(
            f"{embedding_name}: kết quả UMAP 2D không hợp lệ"
        )

    for experiment in UMAP_2D_EXPERIMENTS:
        group = assignments.loc[
            assignments["experiment"] == experiment
        ].copy()

        if group["article_id"].duplicated().any():
            raise ValueError(
                f"{embedding_name}, {experiment}: "
                "article_id bị trùng"
            )

        if (
            len(group) != len(article_ids)
            or set(group["article_id"]) != set(article_ids)
        ):
            raise ValueError(
                f"{embedding_name}, {experiment}: "
                "article_id không khớp artifact"
            )

        # Sắp xếp nhãn theo đúng thứ tự vector trước khi tô màu.
        group = group.set_index("article_id").loc[article_ids]
        labels = group["cluster_id"].to_numpy(dtype=int)

        cluster_sizes = (
            pd.Series(labels[labels != -1])
            .value_counts()
        )
        highlighted = cluster_sizes.head(8).index.tolist()

        noise_mask = labels == -1
        highlighted_mask = np.isin(
            labels,
            highlighted,
        )
        other_mask = (
            ~noise_mask & ~highlighted_mask
        )

        fig, ax = plt.subplots(
            figsize=(10, 7),
            constrained_layout=True,
        )

        ax.scatter(
            points[other_mask, 0],
            points[other_mask, 1],
            s=5,
            color="#D9D9D9",
            alpha=0.45,
            rasterized=True,
            label="Các cụm khác",
        )
        ax.scatter(
            points[noise_mask, 0],
            points[noise_mask, 1],
            s=6,
            color="#777777",
            alpha=0.55,
            rasterized=True,
            label=(
                f"Nhiễu (-1): "
                f"{int(noise_mask.sum())} bài"
            ),
        )

        palette = plt.get_cmap("tab10")

        for index, cluster_id in enumerate(highlighted):
            mask = labels == cluster_id

            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=8,
                color=palette(index),
                alpha=0.8,
                rasterized=True,
                label=(
                    f"Cụm {cluster_id}: "
                    f"{int(mask.sum())} bài"
                ),
            )

        ax.set_title(
            f"{LABELS[embedding_name]} | {experiment}\n"
            f"{len(cluster_sizes)} cụm; "
            f"{int(noise_mask.sum())} bài nhiễu"
        )
        ax.set_xlabel("UMAP chiều 1")
        ax.set_ylabel("UMAP chiều 2")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            loc="best",
            fontsize=8,
            markerscale=2,
            frameon=True,
        )

        model = embedding_name.split("_", 1)[0]
        output_path = (
            figures_dir
            / f"m11_{model}_{experiment}_umap2d.png"
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )
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

    embedding_names = list(
        config["embedding_artifacts"]
    )
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

    if (
        actual != expected
        or len(summary) != len(expected)
    ):
        raise ValueError(
            "Summary thiếu cấu hình hoặc có dòng bị trùng"
        )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Hai hình tổng quan cho đủ 32 run.
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

    # Bốn hình phân bố kích thước cụm.
    for embedding_name in embedding_names:
        assignments = pd.read_csv(
            output_dir
            / f"{embedding_name}_assignments.csv",
            encoding="utf-8-sig",
            dtype={"article_id": str},
        )

        plot_cluster_sizes(
            assignments,
            figures_dir
            / SIZE_FIGURE_NAMES[embedding_name],
            embedding_name,
        )

        # Chỉ vẽ 2D cho hai embedding title + content.
        if embedding_name in UMAP_2D_EMBEDDINGS:
            embedding_table = load_embedding_artifact(
                ROOT
                / config["embedding_artifacts"][
                    embedding_name
                ]
            )

            plot_umap_2d(
                embedding_name,
                embedding_table,
                assignments,
                config["umap"],
                figures_dir,
            )


if __name__ == "__main__":
    main()