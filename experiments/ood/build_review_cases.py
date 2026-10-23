"""Tạo bản nháp các ca đọc định tính cho E07."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ASSIGNMENTS_PATH = (
    ROOT / "results/hdbscan/ood/ood_assignments.csv"
)
ARTICLES_PATH = (
    ROOT / "data/ood/oct2026/news_articles_p0.csv"
)
OUTPUT_PATH = (
    ROOT / "results/hdbscan/ood/ood_reviewed_cases.csv"
)

# article_id, cluster_id dự kiến, nhãn nhận xét, bằng chứng,

CASES = [
    (
        "VNE_823fdaf25ed8",
        16,
        "coherent_assignment",
        "Bài thuật phản ứng của Guardiola trước phán quyết "
        "về vi phạm tài chính của Man City.",
        "Cùng vụ việc với VNE_ddbb22df495b trong cụm 16.",
    ),
    (
        "VNE_ddbb22df495b",
        16,
        "coherent_assignment",
        "Bài giải thích nội dung phán quyết đối với Man City.",
        "Cụm 16 nối hai góc đưa tin về cùng vụ việc.",
    ),
    (
        "VNE_00383be52a24",
        94,
        "coherent_assignment",
        "Bài nói VN-Index giảm và mất mốc 1.700 điểm.",
        "Phù hợp với nhóm bài về diễn biến thị trường "
        "chứng khoán trong cụm 94.",
    ),
    (
        "TTR_c1b5db2eb386",
        94,
        "coherent_assignment",
        "Bài phân tích xu hướng giảm của VN-Index.",
        "Cùng chủ đề thị trường chứng khoán với "
        "VNE_00383be52a24 dù khác nguồn báo.",
    ),
    (
        "TTR_096a819aaba9",
        1,
        "broad_cluster",
        "Bài thuật ca phẫu thuật lấy dị vật khỏi cổ trẻ em.",
        "Thuộc miền sức khỏe nhưng khác bệnh và sự kiện "
        "với các bài phòng dịch trong cụm 1.",
    ),
    (
        "VNE_0e645966f432",
        1,
        "broad_cluster",
        "Bài khuyến cáo phòng bệnh khi số ca cúm A "
        "và viêm hô hấp tăng.",
        "Cùng miền sức khỏe với TTR_096a819aaba9 "
        "nhưng không cùng sự kiện cụ thể.",
    ),
    (
        "TTR_a28f3d6c0095",
        21,
        "broad_cluster",
        "Bài bàn về chi tiêu AI và hạ tầng dữ liệu "
        "của doanh nghiệp.",
        "Cụm 21 gom bài AI thuộc những ứng dụng "
        "và bối cảnh khác nhau.",
    ),
    (
        "TTR_e948dcd67660",
        21,
        "broad_cluster",
        "Bài trình bày cách dùng AI cải thiện "
        "dự báo thời tiết.",
        "Cùng chủ đề AI với TTR_a28f3d6c0095 "
        "nhưng khác ứng dụng và sự kiện.",
    ),
    (
        "VNE_02d57a473b01",
        -1,
        "valid_topic_outlier",
        "Bài đưa tin GDP quý III tăng 9,95% "
        "theo số liệu của Cục Thống kê.",
        "Bài có chủ đề rõ nhưng được gán -1; "
        "không xem -1 là tin lỗi.",
    ),
    (
        "TTR_a73b15048ba0",
        -1,
        "valid_topic_outlier",
        "Bài từ nguồn khác cũng đưa tin "
        "GDP quý III tăng 9,95%.",
        "Cùng sự kiện kinh tế với VNE_02d57a473b01 "
        "nhưng cả hai đều được gán -1.",
    ),
    (
        "VNE_d53a00b1d212",
        -1,
        "possible_missed_assignment",
        "Bài nói về người cung cấp tài liệu "
        "dẫn tới cuộc điều tra Man City.",
        "Liên quan vụ việc trong cụm 16 nhưng trọng tâm "
        "là an toàn nhân chứng; đây là ca ranh giới, "
        "chưa khẳng định mô hình gán sai.",
    ),
    (
        "VNN_7a6a329eb7e0",
        -1,
        "uncertain_noise",
        "Bài thuật một vụ khởi tố cụ thể ở Ninh Bình.",
        "Sự kiện riêng lẻ có thể phù hợp với nhãn -1; "
        "cần đối chiếu thêm các cụm pháp luật gần nó.",
    ),
]


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH} đã tồn tại. "
            "Không ghi đè nhận xét bạn đã sửa."
        )

    for path in (ASSIGNMENTS_PATH, ARTICLES_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    assignments = pd.read_csv(
        ASSIGNMENTS_PATH,
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )
    articles = pd.read_csv(
        ARTICLES_PATH,
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )

    if assignments["article_id"].duplicated().any():
        raise ValueError("Assignments có article_id trùng.")
    if articles["article_id"].duplicated().any():
        raise ValueError("Dữ liệu P0 có article_id trùng.")

    assignments = assignments.set_index("article_id")
    articles = articles.set_index("article_id")

    rows = []

    for (
        article_id,
        expected_cluster,
        review_label,
        evidence,
        interpretation,
    ) in CASES:
        if article_id not in assignments.index:
            raise ValueError(
                f"Không tìm thấy {article_id} trong assignments."
            )
        if article_id not in articles.index:
            raise ValueError(
                f"Không tìm thấy {article_id} trong dữ liệu P0."
            )

        assignment = assignments.loc[article_id]
        article = articles.loc[article_id]
        actual_cluster = int(assignment["cluster_id"])

        if actual_cluster != expected_cluster:
            raise ValueError(
                f"{article_id}: dự kiến cụm "
                f"{expected_cluster}, thực tế {actual_cluster}."
            )

        if not str(article["processed_content"]).strip():
            raise ValueError(
                f"{article_id} không có nội dung P0 để đọc."
            )

        rows.append(
            {
                "article_id": article_id,
                "cluster_id": actual_cluster,
                "source": assignment["source"],
                "title": assignment["title"],
                "url": assignment["url"],
                "review_label": review_label,
                "evidence": evidence,
                "interpretation": interpretation,
                "review_status": "draft_needs_tv4_review",
            }
        )

    output = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Đã tạo {len(output)} ca nháp: {OUTPUT_PATH}")
    print(
        "Đọc lại từng bài rồi đổi review_status thành "
        "'confirmed' cho các dòng bạn đồng ý."
    )


if __name__ == "__main__":
    main()