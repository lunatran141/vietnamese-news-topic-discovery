"""Vẽ các biểu đồ giải thích kết quả BERTopic."""

from pathlib import Path
from typing import Callable
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import matplotlib

# Dùng backend không cần mở cửa sổ giao diện.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SELECTED_RUN = "e1_vietnamese_sbert"



RUN_COLORS = {
    "e0": "#B2E89D",  # xanh
    "e1": "#8EDBE5",  # Xanh cyan
    "e2": "#F5B5D2",  # Hồng
    "e3": "#C5B3ED",  # Tím lavender
    "e4": "#FFD3A5",  # Cam đào
}

TOPIC_COLORS = [
    "#79CFE0",  # Xanh cyan
    "#F3A6C8",  # Hồng phấn
    "#B9A3E3",  # Tím lavender
    "#8FB9E8",  # Xanh dương
    "#E89AC7",  # Hồng tím
    "#9E9AD9",  # Tím xanh
    "#9DD9C5",  # Xanh mint
    "#F6B89E",  # Cam hồng
    "#C9B4E4",  # Tím nhạt
    "#F4C2D7",  # Hồng sáng
    "#A8D8EA",  # Xanh da trời
    "#FFD3A5",  # Cam đào
    "#C7E9B0",  # Xanh lá pastel
    "#F7DC9C",  # Vàng kem
    "#D6B5E8",  # Tím hoa cà
]

# Outlier cần tối hơn để phân biệt với các topic.
OUTLIER_POINT_COLOR = "#616875"

# Các topic nhỏ không thuộc top 15
OTHER_TOPIC_COLOR = "#E2E5EC"

SELECTED_COLOR = RUN_COLORS["e1"]
OTHER_COLOR = "#B0BEC5"
OUTLIER_COLOR = "#E53935"

def validate_columns(
    data: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """Kiểm tra bảng có đủ các cột cần để vẽ."""
    missing_columns = required_columns.difference(
        data.columns
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            f"Thiếu cột để vẽ biểu đồ: {missing_text}"
        )


def get_run_labels(
    summary: pd.DataFrame,
) -> list[str]:
    """Rút gọn run_id thành E0–E4."""
    return [
        str(run_id).split("_", maxsplit=1)[0].upper()
        for run_id in summary["run_id"]
    ]


def get_run_colors(
    summary: pd.DataFrame,
) -> list[str]:
    """Gán một màu nhất quán cho từng cấu hình E0–E4."""
    colors = []

    for run_id in summary["run_id"]:
        short_id = str(run_id).split(
            "_",
            maxsplit=1,
        )[0].lower()

        colors.append(
            RUN_COLORS.get(short_id, OTHER_COLOR)
        )

    return colors

def add_experiment_legend(
    figure: plt.Figure,
) -> None:
    """Giải thích màu và ý nghĩa của từng cấu hình."""
    legend_items = [
        Patch(
            facecolor=RUN_COLORS["e0"],
            label="E0: Baseline MiniLM",
        ),
        Patch(
            facecolor=RUN_COLORS["e1"],
            label="E1: Vietnamese SBERT",
        ),
        Patch(
            facecolor=RUN_COLORS["e2"],
            label="E2: E1 + UMAP n_neighbors=30",
        ),
        Patch(
            facecolor=RUN_COLORS["e3"],
            label="E3: E1 + HDBSCAN min_cluster_size=30",
        ),
        Patch(
            facecolor=RUN_COLORS["e4"],
            label="E4: E1 + chỉ dùng tiêu đề",
        ),
    ]

    figure.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=True,
        fontsize=9,
    )

def add_bar_labels(
    axis: plt.Axes,
    formatter: Callable[[float], str],
) -> None:
    """Ghi giá trị lên đầu mỗi cột."""
    for bar in axis.patches:
        value = float(bar.get_height())

        axis.annotate(
            formatter(value),
            (
                bar.get_x() + bar.get_width() / 2,
                value,
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_experiment_dashboard(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """So sánh năm cấu hình theo các chỉ số chính."""
    required_columns = {
        "run_id",
        "topic_count",
        "validation_outlier_rate",
        "topic_diversity",
        "runtime_seconds",
    }
    validate_columns(summary, required_columns)

    plot_data = summary.copy()

    numeric_columns = required_columns - {"run_id"}

    for column in numeric_columns:
        plot_data[column] = pd.to_numeric(
            plot_data[column],
            errors="raise",
        )

    labels = get_run_labels(plot_data)
    colors = get_run_colors(plot_data)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    panels = [
        (
            axes[0, 0],
            "topic_count",
            "Số topic",
            "Số lượng",
            lambda value: f"{value:.0f}",
        ),
        (
            axes[0, 1],
            "validation_outlier_rate",
            "Outlier trên validation",
            "Tỷ lệ",
            lambda value: f"{value:.1%}",
        ),
        (
            axes[1, 0],
            "topic_diversity",
            "Độ đa dạng từ khóa",
            "Topic diversity",
            lambda value: f"{value:.3f}",
        ),
        (
            axes[1, 1],
            "runtime_seconds",
            "Thời gian chạy",
            "Giây",
            lambda value: f"{value:.0f}s",
        ),
    ]

    for (
        axis,
        column,
        title,
        ylabel,
        formatter,
    ) in panels:
        values = plot_data[column]

        axis.bar(
            labels,
            values,
            color=colors,
        )
        axis.set_title(
            title,
            fontweight="bold",
            pad=14,
        )
        axis.set_xlabel("Cấu hình")
        axis.set_ylabel(ylabel)
        maximum_value = float(values.max())

        if maximum_value > 0:
            axis.set_ylim(
                0,
                maximum_value * 1.25,
            )

        axis.grid(
            axis="y",
            linestyle="--",
            alpha=0.3,
        )
        add_bar_labels(
            axis,
            formatter,
        )

    figure.suptitle(
        "So sánh các cấu hình BERTopic\n"
        "E0: baseline | E1: Vietnamese SBERT | "
        "E2: thay đổi UMAP | E3: thay đổi HDBSCAN | "
        "E4: chỉ tiêu đề",
        fontsize=15,
        fontweight="bold",
    )

    add_experiment_legend(figure)
    
    figure.tight_layout(
        rect=(0, 0, 1, 0.88)
    )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_validation_outlier(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """So sánh tỷ lệ outlier trên validation."""
    validate_columns(
        summary,
        {
            "run_id",
            "validation_outlier_rate",
        },
    )

    plot_data = summary.copy()

    plot_data["validation_outlier_rate"] = (
        pd.to_numeric(
            plot_data["validation_outlier_rate"],
            errors="raise",
        )
    )

    labels = get_run_labels(plot_data)
    colors = get_run_colors(plot_data)

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.bar(
        labels,
        plot_data["validation_outlier_rate"],
        color=colors,
    )

    axis.set_title(
        "Tỷ lệ bài validation "
        "không thuộc cụm rõ ràng",
        fontweight="bold",
    )
    axis.set_xlabel("Cấu hình")
    axis.set_ylabel("Outlier rate")

    maximum_value = max(
        plot_data["validation_outlier_rate"]
    )
    axis.set_ylim(0, maximum_value * 1.2)

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    add_bar_labels(
        axis,
        lambda value: f"{value:.1%}",
    )

    axis.text(
        0.5,
        -0.16,
        "Giá trị thấp hơn cho thấy nhiều bài "
        "validation được gán vào topic hơn.",
        transform=axis.transAxes,
        ha="center",
        fontsize=10,
    )

    add_experiment_legend(figure)

    figure.tight_layout(
        rect=(0, 0.07, 1, 0.86)
    )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_topic_diversity(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """So sánh độ đa dạng từ khóa giữa các cấu hình."""
    validate_columns(
        summary,
        {
            "run_id",
            "topic_diversity",
        },
    )

    plot_data = summary.copy()

    plot_data["topic_diversity"] = pd.to_numeric(
        plot_data["topic_diversity"],
        errors="raise",
    )

    labels = get_run_labels(plot_data)
    colors = get_run_colors(plot_data)

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.bar(
        labels,
        plot_data["topic_diversity"],
        color=colors,
    )

    axis.set_title(
        "Độ đa dạng từ khóa giữa các topic",
        fontweight="bold",
    )
    axis.set_xlabel("Cấu hình")
    axis.set_ylabel("Topic diversity")
    axis.set_ylim(0, 1)

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    add_bar_labels(
        axis,
        lambda value: f"{value:.3f}",
    )

    axis.text(
        0.5,
        -0.16,
        "Giá trị cao cho thấy từ khóa giữa "
        "các topic ít bị lặp lại hơn.",
        transform=axis.transAxes,
        ha="center",
        fontsize=10,
    )

    add_experiment_legend(figure)

    figure.tight_layout(
        rect=(0, 0.07, 1, 0.86)
    )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_title_content_ablation(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """So sánh BERTopic chỉ tiêu đề với tiêu đề + nội dung trên validation."""
    required = {
        "run_id",
        "input_text_label",
        "topic_count",
        "validation_outlier_rate",
        "topic_diversity",
        "validation_adjusted_rand_index",
        "validation_normalized_mutual_info",
        "validation_npmi_coherence",
    }
    validate_columns(summary, required)

    selected = summary.loc[
        summary["input_text_label"].isin(
            ["title_only", "title_content"]
        )
    ].copy()
    selected = selected.drop_duplicates(
        subset=["input_text_label"], keep="last"
    ).set_index("input_text_label")

    missing = {"title_only", "title_content"}.difference(selected.index)
    if missing:
        raise ValueError(
            "Chưa đủ kết quả ablation BERTopic: "
            + ", ".join(sorted(missing))
        )

    selected = selected.loc[["title_only", "title_content"]]
    numeric_columns = list(required.difference({"run_id", "input_text_label"}))
    selected[numeric_columns] = selected[numeric_columns].apply(
        pd.to_numeric, errors="raise"
    )

    labels = ["Chỉ tiêu đề", "Tiêu đề + nội dung"]
    # Pastel tươi, đủ tương phản khi chiếu và khi in trên nền trắng.
    colors = ["#FFB7C5", "#74D7C4"]
    metric_columns = [
        "validation_adjusted_rand_index",
        "validation_normalized_mutual_info",
        "validation_npmi_coherence",
        "topic_diversity",
    ]
    metric_labels = ["ARI", "NMI", "NPMI", "Diversity"]
    positions = np.arange(len(metric_columns))
    width = 0.34

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for index, (variant, label, color) in enumerate(
        zip(selected.index, labels, colors)
    ):
        values = 100 * selected.loc[variant, metric_columns].to_numpy(dtype=float)
        bars = axes[0].bar(
            positions + (index - 0.5) * width,
            values,
            width,
            label=label,
            color=color,
        )
        for bar, value in zip(bars, values):
            axes[0].annotate(
                f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4 if value >= 0 else -13),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    axes[0].axhline(0, color="#616875", linewidth=0.8)
    axes[0].set_xticks(positions, metric_labels)
    axes[0].set_ylabel("Điểm số (×100)")
    axes[0].set_title("Chất lượng chủ đề trên validation", fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend()

    outlier_values = 100 * selected["validation_outlier_rate"].to_numpy(dtype=float)
    bars = axes[1].bar(labels, outlier_values, color=colors, width=0.58)
    axes[1].set_ylabel("Outlier validation (%)")
    axes[1].set_title("Độ bao phủ và số topic", fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    for bar, outlier, topic_count in zip(
        bars,
        outlier_values,
        selected["topic_count"].to_numpy(dtype=int),
    ):
        axes[1].annotate(
            f"{outlier:.1f}% outlier\n{topic_count} topic",
            (bar.get_x() + bar.get_width() / 2, outlier),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    figure.suptitle(
        "Ablation đầu vào BERTopic: Title only và Title + Content",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Hai run dùng cùng Vietnamese SBERT, UMAP và HDBSCAN; chỉ thay trường văn bản đầu vào.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def format_topic_name(name: str) -> str:
    """Chuyển tên topic thành nhãn ngắn dễ đọc."""
    parts = str(name).split("_")

    if (
        parts
        and parts[0].lstrip("-").isdigit()
    ):
        parts = parts[1:]

    return ", ".join(parts[:4])


def plot_topic_sizes(
    topic_info: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """Vẽ kích thước các topic lớn nhất."""
    validate_columns(
        topic_info,
        {
            "Topic",
            "Count",
            "Name",
        },
    )

    if top_n <= 0:
        raise ValueError(
            "top_n phải lớn hơn 0."
        )

    plot_data = topic_info.copy()

    plot_data["Topic"] = pd.to_numeric(
        plot_data["Topic"],
        errors="raise",
    )
    plot_data["Count"] = pd.to_numeric(
        plot_data["Count"],
        errors="raise",
    )

    outlier_rows = plot_data.loc[
        plot_data["Topic"] == -1
    ]

    if outlier_rows.empty:
        outlier_count = 0
    else:
        outlier_count = int(
            outlier_rows["Count"].sum()
        )

    plot_data = (
        plot_data.loc[
            plot_data["Topic"] != -1
        ]
        .nlargest(top_n, "Count")
        .sort_values("Count")
        .copy()
    )

    plot_data["label"] = plot_data.apply(
        lambda row: (
            f"Topic {int(row['Topic'])}: "
            f"{format_topic_name(row['Name'])}"
        ),
        axis=1,
    )

    figure, axis = plt.subplots(
        figsize=(12, 8)
    )

    bars = axis.barh(
        plot_data["label"],
        plot_data["Count"],
        color=SELECTED_COLOR,
    )

    axis.set_title(
        f"{top_n} topic lớn nhất "
        "của cấu hình E1",
        fontweight="bold",
    )
    axis.set_xlabel("Số bài train")
    axis.set_ylabel("")

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.3,
    )

    for bar in bars:
        value = int(bar.get_width())

        axis.annotate(
            str(value),
            (
                value,
                bar.get_y()
                + bar.get_height() / 2,
            ),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    axis.text(
        0.99,
        0.02,
        "Outlier không hiển thị "
        f"trong biểu đồ: {outlier_count} bài",
        transform=axis.transAxes,
        ha="right",
        color=OUTLIER_COLOR,
        fontsize=10,
    )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

def plot_embedding_map(
    projection: pd.DataFrame,
    topic_summary: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 8,
) -> None:
    """Vẽ bản đồ embedding 2D và các cụm topic nổi bật."""
    validate_columns(
        projection,
        {
            "article_id",
            "split",
            "topic",
            "x",
            "y",
            "is_outlier",
        },
    )
    validate_columns(
        topic_summary,
        {
            "topic",
            "topic_name",
            "document_count",
        },
    )

    if top_n <= 0:
        raise ValueError("top_n phải lớn hơn 0.")

    plot_data = projection.copy()
    summary_data = topic_summary.copy()

    for column in ["topic", "x", "y"]:
        plot_data[column] = pd.to_numeric(
            plot_data[column],
            errors="raise",
        )

    summary_data["topic"] = pd.to_numeric(
        summary_data["topic"],
        errors="raise",
    )
    summary_data["document_count"] = pd.to_numeric(
        summary_data["document_count"],
        errors="raise",
    )

    selected_topics = (
        summary_data.loc[summary_data["topic"] != -1]
        .nlargest(top_n, "document_count")
        .copy()
    )

    topic_ids = selected_topics["topic"].astype(int).tolist()

    topic_color_map = {
        topic: TOPIC_COLORS[index % len(TOPIC_COLORS)]
        for index, topic in enumerate(topic_ids)
    }

    topic_name_map = {
        int(row["topic"]): str(row["topic_name"])
        for _, row in selected_topics.iterrows()
    }

    figure, axis = plt.subplots(
        figsize=(14, 9)
    )

    # Các topic nhỏ không thuộc top_n.
    other_topics = plot_data.loc[
        (plot_data["topic"] != -1)
        & (~plot_data["topic"].isin(topic_ids))
    ]

    axis.scatter(
        other_topics["x"],
        other_topics["y"],
        color=OTHER_TOPIC_COLOR,
        s=14,
        alpha=0.35,
        label="Các topic nhỏ khác",
    )

    # Outlier do HDBSCAN không gán vào cụm rõ ràng.
    outliers = plot_data.loc[
        plot_data["topic"] == -1
    ]

    axis.scatter(
        outliers["x"],
        outliers["y"],
        color=OUTLIER_POINT_COLOR,
        marker="x",
        s=16,
        alpha=0.32,
        label="Outlier (-1)",
    )

    split_markers = {
        "train": "o",
        "val": "^",
        "validation": "^",
    }

    for topic in topic_ids:
        topic_data = plot_data.loc[
            plot_data["topic"] == topic
        ]

        for split, split_data in topic_data.groupby("split"):
            marker = split_markers.get(
                str(split).lower(),
                "o",
            )

            axis.scatter(
                split_data["x"],
                split_data["y"],
                color=topic_color_map[topic],
                marker=marker,
                s=28,
                alpha=0.70,
                edgecolors="white",
                linewidths=0.25,
            )

        if not topic_data.empty:
            center_x = topic_data["x"].median()
            center_y = topic_data["y"].median()

            axis.annotate(
                f"T{topic}",
                (center_x, center_y),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": topic_color_map[topic],
                    "alpha": 0.85,
                },
            )

    legend_items = []

    for topic in topic_ids:
        topic_name = format_topic_name(
            topic_name_map[topic]
        )

        legend_items.append(
            Patch(
                facecolor=topic_color_map[topic],
                label=f"T{topic}: {topic_name}",
            )
        )

    legend_items.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#777777",
                label="Train",
                markersize=7,
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor="#777777",
                label="Validation",
                markersize=7,
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color=OUTLIER_POINT_COLOR,
                linestyle="none",
                label="Outlier HDBSCAN",
                markersize=7,
            ),
        ]
    )

    axis.legend(
        handles=legend_items,
        title="Topic và tập dữ liệu",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=8,
        title_fontsize=10,
        ncol=1,
        borderpad=0.8,
        labelspacing=0.65,
    )

    axis.set_title(
        "Bản đồ embedding 2D của cấu hình E1",
        fontweight="bold",
        pad=14,
    )
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")

    axis.text(
        0.5,
        -0.11,
        "Các điểm gần nhau biểu diễn bài báo có "
        "ngữ nghĩa tương đồng; màu biểu diễn topic.",
        transform=axis.transAxes,
        ha="center",
        fontsize=10,
    )

    axis.grid(
        linestyle="--",
        alpha=0.20,
    )

    figure.tight_layout(
        rect=(0, 0.05, 0.74, 1)
    )
    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

def plot_ctfidf_keywords(
    topic_keywords: pd.DataFrame,
    topic_summary: pd.DataFrame,
    output_path: str | Path,
    top_n_topics: int = 6,
    top_n_words: int = 8,
) -> None:
    """Vẽ từ khóa c-TF-IDF của các topic lớn nhất."""
    validate_columns(
        topic_keywords,
        {
            "topic",
            "rank",
            "word",
            "ctfidf_score",
        },
    )
    validate_columns(
        topic_summary,
        {
            "topic",
            "topic_name",
            "document_count",
        },
    )

    if top_n_topics <= 0:
        raise ValueError(
            "top_n_topics phải lớn hơn 0."
        )

    if top_n_words <= 0:
        raise ValueError(
            "top_n_words phải lớn hơn 0."
        )

    keywords = topic_keywords.copy()
    summary = topic_summary.copy()

    keywords["topic"] = pd.to_numeric(
        keywords["topic"],
        errors="raise",
    )
    keywords["rank"] = pd.to_numeric(
        keywords["rank"],
        errors="raise",
    )
    keywords["ctfidf_score"] = pd.to_numeric(
        keywords["ctfidf_score"],
        errors="raise",
    )

    summary["topic"] = pd.to_numeric(
        summary["topic"],
        errors="raise",
    )
    summary["document_count"] = pd.to_numeric(
        summary["document_count"],
        errors="raise",
    )

    largest_topics = (
        summary.loc[summary["topic"] != -1]
        .nlargest(top_n_topics, "document_count")
        .copy()
    )

    number_of_topics = len(largest_topics)
    number_of_columns = 3
    number_of_rows = (
        number_of_topics + number_of_columns - 1
    ) // number_of_columns

    figure, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(17, 4.6 * number_of_rows),
        squeeze=False,
    )

    flat_axes = axes.flatten()

    for index, (_, topic_row) in enumerate(
        largest_topics.iterrows()
    ):
        axis = flat_axes[index]
        topic = int(topic_row["topic"])

        topic_words = (
            keywords.loc[
                keywords["topic"] == topic
            ]
            .nsmallest(top_n_words, "rank")
            .sort_values("ctfidf_score")
            .copy()
        )

        bars = axis.barh(
            topic_words["word"],
            topic_words["ctfidf_score"],
            color=TOPIC_COLORS[
                index % len(TOPIC_COLORS)
            ],
            alpha=0.85,
        )

        short_name = format_topic_name(
            topic_row["topic_name"]
        )

        axis.set_title(
            f"T{topic}: {short_name}\n"
            f"{int(topic_row['document_count'])} bài",
            fontweight="bold",
            fontsize=10,
        )
        axis.set_xlabel("Điểm c-TF-IDF")
        axis.set_ylabel("")
        axis.grid(
            axis="x",
            linestyle="--",
            alpha=0.25,
        )

        for bar in bars:
            score = float(bar.get_width())

            axis.annotate(
                f"{score:.3f}",
                (
                    score,
                    bar.get_y()
                    + bar.get_height() / 2,
                ),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
            )

        maximum_score = topic_words[
            "ctfidf_score"
        ].max()

        if pd.notna(maximum_score):
            axis.set_xlim(
                0,
                float(maximum_score) * 1.22,
            )
            axis.xaxis.set_major_locator(
                MaxNLocator(nbins=5, min_n_ticks=4)
            )
            axis.xaxis.set_major_formatter(
                FormatStrFormatter(
                    "%.3f" if float(maximum_score) < 0.1 else "%.2f"
                )
            )
            axis.tick_params(axis="x", labelsize=8)

    # Ẩn các ô thừa nếu số topic không chia hết cho 3.
    for index in range(
        number_of_topics,
        len(flat_axes),
    ):
        flat_axes[index].set_visible(False)

    figure.suptitle(
        "Từ khóa đại diện của các topic theo c-TF-IDF",
        fontsize=16,
        fontweight="bold",
    )

    figure.text(
        0.5,
        0.01,
        "Điểm cao cho thấy từ khóa xuất hiện nổi bật "
        "trong topic nhưng ít phổ biến ở các topic khác.",
        ha="center",
        fontsize=10,
    )

    figure.tight_layout(
        rect=(0, 0.04, 1, 0.94),
        h_pad=2.2,
        w_pad=2.5,
    )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

def plot_bertopic_pipeline(
    output_path: str | Path,
) -> None:
    """Vẽ sơ đồ cân đối các bước trong pipeline BERTopic."""
    figure, axis = plt.subplots(
        figsize=(18, 6)
    )

    axis.set_xlim(0, 18)
    axis.set_ylim(0, 6)
    axis.axis("off")

    steps = [
        {
            "title": "1. Dataset\nđã đóng băng",
            "detail": "Train: 6.493 bài\nValidation: 1.391 bài",
            "color": "#DCE6F2",
        },
        {
            "title": "2. Tạo văn bản\nđầu vào",
            "detail": "Tiêu đề + nội dung\nGiữ đúng thứ tự bài",
            "color": "#F4C2D7",
        },
        {
            "title": "3. Embedding",
            "detail": "Vietnamese SBERT\nVăn bản → vector ngữ nghĩa",
            "color": "#9EDBE5",
        },
        {
            "title": "4. UMAP",
            "detail": "Giảm số chiều\nGiữ cấu trúc lân cận",
            "color": "#C5B3ED",
        },
        {
            "title": "5. HDBSCAN",
            "detail": "Phân cụm mật độ\nPhát hiện outlier (-1)",
            "color": "#F6B89E",
        },
        {
            "title": "6. c-TF-IDF",
            "detail": "Tìm từ khóa đặc trưng\nĐặt tên và mô tả topic",
            "color": "#A6D8C5",
        },
    ]

    box_width = 2.35
    box_height = 2.55
    box_y = 1.85

    left_margin = 0.55
    horizontal_step = 2.95

    for index, step in enumerate(steps):
        box_x = (
            left_margin
            + index * horizontal_step
        )

        box = FancyBboxPatch(
            (box_x, box_y),
            box_width,
            box_height,
            boxstyle="round,pad=0.10",
            facecolor=step["color"],
            edgecolor="#667085",
            linewidth=1.3,
        )
        axis.add_patch(box)

        axis.text(
            box_x + box_width / 2,
            box_y + 1.75,
            step["title"],
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            linespacing=1.25,
        )

        axis.text(
            box_x + box_width / 2,
            box_y + 0.77,
            step["detail"],
            ha="center",
            va="center",
            fontsize=8.8,
            linespacing=1.45,
        )

        if index < len(steps) - 1:
            next_box_x = (
                left_margin
                + (index + 1) * horizontal_step
            )

            axis.annotate(
                "",
                xy=(
                    next_box_x - 0.08,
                    box_y + box_height / 2,
                ),
                xytext=(
                    box_x + box_width + 0.08,
                    box_y + box_height / 2,
                ),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#667085",
                    "linewidth": 1.8,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )

    axis.set_title(
        "Pipeline khám phá chủ đề tin tức "
        "tiếng Việt bằng BERTopic",
        fontsize=18,
        fontweight="bold",
        pad=22,
    )

    axis.text(
        9,
        0.85,
        "Đầu ra: topic của từng bài  •  xác suất gán topic  •  "
        "từ khóa c-TF-IDF  •  chỉ số đánh giá  •  biểu đồ trực quan",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#475467",
    )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

def plot_cluster_evaluation(
    document_topics: pd.DataFrame,
    topic_summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Vẽ dashboard đánh giá độ cân bằng và độ rõ của cụm."""
    validate_columns(
        document_topics,
        {
            "split",
            "topic",
            "topic_probability",
            "is_outlier",
        },
    )
    validate_columns(
        topic_summary,
        {
            "topic",
            "topic_name",
            "document_count",
        },
    )

    documents = document_topics.copy()
    summary = topic_summary.copy()

    documents["topic"] = pd.to_numeric(
        documents["topic"],
        errors="raise",
    )
    documents["topic_probability"] = pd.to_numeric(
        documents["topic_probability"],
        errors="coerce",
    )

    if documents["is_outlier"].dtype != bool:
        documents["is_outlier"] = (
            documents["is_outlier"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                }
            )
        )

    summary["topic"] = pd.to_numeric(
        summary["topic"],
        errors="raise",
    )
    summary["document_count"] = pd.to_numeric(
        summary["document_count"],
        errors="raise",
    )

    valid_summary = summary.loc[
        summary["topic"] != -1
    ].copy()

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10),
    )

    # 1. Phân phối kích thước các topic.
    topic_sizes = valid_summary["document_count"]

    axes[0, 0].hist(
        topic_sizes,
        bins=min(15, len(topic_sizes)),
        color="#79CFE0",
        edgecolor="white",
        alpha=0.85,
    )

    median_size = float(topic_sizes.median())

    axes[0, 0].axvline(
        median_size,
        color="#A855A8",
        linestyle="--",
        linewidth=1.8,
        label=f"Trung vị: {median_size:.0f} bài",
    )

    axes[0, 0].set_title(
        "Phân phối kích thước topic",
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("Số bài trong topic")
    axes[0, 0].set_ylabel("Số topic")
    axes[0, 0].legend()
    axes[0, 0].grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    # 2. Các topic lớn nhất.
    largest_topics = (
        valid_summary.nlargest(
            10,
            "document_count",
        )
        .sort_values("document_count")
        .copy()
    )

    largest_topics["label"] = largest_topics.apply(
        lambda row: (
            f"T{int(row['topic'])}: "
            f"{format_topic_name(row['topic_name'])}"
        ),
        axis=1,
    )

    bars = axes[0, 1].barh(
        largest_topics["label"],
        largest_topics["document_count"],
        color=TOPIC_COLORS[: len(largest_topics)],
        alpha=0.85,
    )

    axes[0, 1].set_title(
        "10 topic có nhiều bài nhất",
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("Số bài train")
    axes[0, 1].grid(
        axis="x",
        linestyle="--",
        alpha=0.25,
    )

    for bar in bars:
        value = int(bar.get_width())

        axes[0, 1].annotate(
            str(value),
            (
                value,
                bar.get_y()
                + bar.get_height() / 2,
            ),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    maximum_count = largest_topics[
        "document_count"
    ].max()

    axes[0, 1].set_xlim(
        0,
        float(maximum_count) * 1.18,
    )

    # 3. Độ tin cậy của các bài đã được gán topic.
    assigned_documents = documents.loc[
        ~documents["is_outlier"]
        & documents["topic_probability"].notna()
    ].copy()

    split_order = [
        split
        for split in ["train", "val", "validation"]
        if split in assigned_documents["split"].unique()
    ]

    probability_groups = [
        assigned_documents.loc[
            assigned_documents["split"] == split,
            "topic_probability",
        ]
        for split in split_order
    ]

    boxplot = axes[1, 0].boxplot(
        probability_groups,
        tick_labels=[
            "Validation"
            if split in {"val", "validation"}
            else "Train"
            for split in split_order
        ],
        patch_artist=True,
        showfliers=False,
    )

    box_colors = [
        "#79CFE0",
        "#F3A6C8",
    ]

    for patch, color in zip(
        boxplot["boxes"],
        box_colors,
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    axes[1, 0].set_title(
        "Độ tin cậy khi gán bài vào topic",
        fontweight="bold",
    )
    axes[1, 0].set_ylabel("Topic probability")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    # 4. Tỷ lệ outlier theo tập dữ liệu.
    outlier_rates = (
        documents.groupby("split")["is_outlier"]
        .mean()
    )

    display_labels = [
        "Validation"
        if split in {"val", "validation"}
        else "Train"
        for split in outlier_rates.index
    ]

    outlier_bars = axes[1, 1].bar(
        display_labels,
        outlier_rates.values,
        color=[
            "#79CFE0",
            "#F3A6C8",
        ][: len(outlier_rates)],
        alpha=0.85,
    )

    axes[1, 1].set_title(
        "Tỷ lệ outlier theo tập dữ liệu",
        fontweight="bold",
    )
    axes[1, 1].set_ylabel("Tỷ lệ")
    axes[1, 1].set_ylim(
        0,
        max(float(outlier_rates.max()) * 1.25, 0.1),
    )
    axes[1, 1].grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    for bar in outlier_bars:
        value = float(bar.get_height())

        axes[1, 1].annotate(
            f"{value:.1%}",
            (
                bar.get_x()
                + bar.get_width() / 2,
                value,
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    figure.suptitle(
        "Đánh giá độ cân bằng và độ rõ của cụm E1",
        fontsize=17,
        fontweight="bold",
    )

    figure.text(
        0.5,
        0.01,
        "Cụm tốt cần có kích thước hợp lý, "
        "xác suất gán cao và tỷ lệ outlier thấp.",
        ha="center",
        fontsize=10,
    )

    figure.tight_layout(
        rect=(0, 0.04, 1, 0.95)
    )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
