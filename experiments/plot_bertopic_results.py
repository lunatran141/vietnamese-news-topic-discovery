"""Tạo các biểu đồ từ kết quả thí nghiệm BERTopic."""

from pathlib import Path

import pandas as pd

from src.visualization.bertopic_plots import (
    plot_experiment_dashboard,
    plot_topic_diversity,
    plot_topic_sizes,
    plot_validation_outlier,
    plot_embedding_map,
    plot_ctfidf_keywords,
    plot_bertopic_pipeline,
    plot_cluster_evaluation,
)


SUMMARY_PATH = Path(
    "results/bertopic_experiments/summary.csv"
)

E1_TOPIC_INFO_PATH = Path(
    "results/bertopic_experiments/"
    "e1_vietnamese_sbert/topic_info.csv"
)

OUTPUT_DIR = Path(
    "results/bertopic_visualization"
)

EMBEDDING_PROJECTION_PATH = Path(
    "results/bertopic_experiments/"
    "e1_vietnamese_sbert/embedding_projection.csv"
)

TOPIC_SUMMARY_PATH = Path(
    "results/bertopic_experiments/"
    "e1_vietnamese_sbert/topic_summary.csv"
)

E1_TOPIC_KEYWORDS_PATH = Path(
    "results/bertopic_experiments/"
    "e1_vietnamese_sbert/topic_keywords.csv"
)

E1_DOCUMENT_TOPICS_PATH = Path(
    "results/bertopic_experiments/"
    "e1_vietnamese_sbert/document_topics.csv"
)

def validate_input_files() -> None:
    """Kiểm tra các file kết quả cần để vẽ."""
    required_files = [
        SUMMARY_PATH,
        E1_TOPIC_INFO_PATH,
        EMBEDDING_PROJECTION_PATH,
        TOPIC_SUMMARY_PATH,
        E1_TOPIC_KEYWORDS_PATH,
        E1_DOCUMENT_TOPICS_PATH,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        missing_text = ", ".join(
            str(path)
            for path in missing_files
        )
        raise FileNotFoundError(
            "Không tìm thấy dữ liệu để vẽ: "
            f"{missing_text}"
        )


def main() -> None:
    """Đọc kết quả và tạo các hình trực quan chính."""
    validate_input_files()

    summary = pd.read_csv(
        SUMMARY_PATH,
        encoding="utf-8-sig",
    )
    topic_info = pd.read_csv(
        E1_TOPIC_INFO_PATH,
        encoding="utf-8-sig",
    )

    embedding_projection = pd.read_csv(
        EMBEDDING_PROJECTION_PATH,
        encoding="utf-8-sig",
    )

    topic_summary = pd.read_csv(
        TOPIC_SUMMARY_PATH,
        encoding="utf-8-sig",
    )
    
    topic_keywords = pd.read_csv(
        E1_TOPIC_KEYWORDS_PATH,
        encoding="utf-8-sig",
    )

    document_topics = pd.read_csv(
        E1_DOCUMENT_TOPICS_PATH,
        encoding="utf-8-sig",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_experiment_dashboard(
        summary,
        OUTPUT_DIR
        / "01_experiment_dashboard.png",
    )

    plot_validation_outlier(
        summary,
        OUTPUT_DIR
        / "02_validation_outlier.png",
    )

    plot_topic_diversity(
        summary,
        OUTPUT_DIR
        / "03_topic_diversity.png",
    )

    plot_topic_sizes(
        topic_info,
        OUTPUT_DIR
        / "04_e1_topic_sizes.png",
        top_n=15,
    )

    plot_embedding_map(
        embedding_projection,
        topic_summary,
        OUTPUT_DIR / "05_e1_embedding_map.png",
        top_n=15,      
    )

    plot_ctfidf_keywords(
        topic_keywords,
        topic_summary,
        OUTPUT_DIR / "06_e1_ctfidf_keywords.png",
        top_n_topics=6,
        top_n_words=8,
    )

    plot_bertopic_pipeline(
        OUTPUT_DIR / "07_bertopic_pipeline.png"
    )

    plot_cluster_evaluation(
        document_topics,
        topic_summary,
        OUTPUT_DIR / "08_e1_cluster_evaluation.png",
    )

    print(
        "Đã tạo các biểu đồ:"
    )

    for image_path in sorted(
        OUTPUT_DIR.glob("*.png")
    ):
        print(f"- {image_path}")


if __name__ == "__main__":
    main()