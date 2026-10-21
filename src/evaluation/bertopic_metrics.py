from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def calculate_outlier_rate(topics: Sequence[int]) -> float:
    """Tính tỷ lệ bài được BERTopic gán vào topic nhiễu -1."""
    if len(topics) == 0:
        raise ValueError("Danh sách topic không được rỗng.")

    outlier_count = sum(topic == -1 for topic in topics)

    return outlier_count / len(topics)


def calculate_topic_diversity(
    topic_words: Mapping[
        int,
        Sequence[tuple[str, float]],
    ],
    top_n: int,
) -> float:
    """Tính tỷ lệ keyword khác nhau giữa các topic."""
    if top_n <= 0:
        raise ValueError("top_n phải lớn hơn 0.")

    selected_words: list[str] = []

    for topic_id, words in topic_words.items():
        if topic_id == -1:
            continue

        selected_words.extend(
            word
            for word, _ in list(words)[:top_n]
        )

    if not selected_words:
        return 0.0

    return len(set(selected_words)) / len(selected_words)


def calculate_external_metrics(
    reference_labels: Sequence[object],
    topics: Sequence[int],
    *,
    ignore_outliers: bool = False,
) -> dict[str, float]:
    """Tính ARI/NMI với nhãn tham chiếu trên cùng một tập bài.

    ``category_gold`` chỉ là chuyên mục báo chí rộng, vì vậy hai chỉ số
    này là phép kiểm tra tham chiếu chứ không phải ground truth topic.
    """
    labels = np.asarray(reference_labels)
    topic_values = np.asarray(topics)

    if labels.ndim != 1 or topic_values.ndim != 1:
        raise ValueError("Nhãn tham chiếu và topic phải là vector một chiều.")
    if len(labels) == 0:
        raise ValueError("Danh sách nhãn không được rỗng.")
    if len(labels) != len(topic_values):
        raise ValueError("Số nhãn tham chiếu phải bằng số topic.")

    mask = np.ones(len(topic_values), dtype=bool)
    if ignore_outliers:
        mask &= topic_values != -1

    labels = labels[mask]
    topic_values = topic_values[mask]
    if len(labels) == 0:
        raise ValueError("Không còn bài nào để tính ARI/NMI.")

    return {
        "adjusted_rand_index": float(
            adjusted_rand_score(labels, topic_values)
        ),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(labels, topic_values)
        ),
    }
