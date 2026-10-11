from collections.abc import Mapping, Sequence


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