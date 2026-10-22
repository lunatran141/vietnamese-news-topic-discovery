
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts/common/lda/lda_k15_seed42"
OUTPUT_DIR = ROOT / "results/lda/topic_label_review"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {
    0: "Giáo dục đại học và tuyển sinh",
    1: "Tài chính, ngân hàng và thị trường tiêu dùng",
    2: "Dinh dưỡng và chăm sóc sức khỏe",
    3: "Thể thao Việt Nam tại ASIAD",
    4: "Tội phạm và điều tra hình sự",
    5: "Giải chạy Marathon và sự kiện chạy bộ",
    6: "Bóng đá trong nước và quốc tế",
    7: "Y tế, bệnh lý và điều trị",
    8: "Thị trường vàng và biến động giá",
    9: "Giáo dục phổ thông và quản lý trường học",
    10: "Thời trang và phong cách người nổi tiếng",
    11: "Âm nhạc và hoạt động biểu diễn",
    12: "Trí tuệ nhân tạo và rủi ro an toàn AI",
    13: "Phát triển doanh nghiệp và công nghệ",
    14: "Quản lý nhà nước và chính sách pháp lý",
}

NOTES = {
    0: "9/10 bài thuộc chủ đề giáo dục đại học; 1 bài Miss World không phù hợp.",
    1: "Topic pha trộn chứng khoán, ngân hàng và iPhone; cần kiểm tra chéo.",
    2: "Các bài liên quan đến dinh dưỡng, thực phẩm và chăm sóc sức khỏe.",
    3: "Cả 10 bài đều liên quan đến ASIAD.",
    4: "Các bài về tội phạm, khởi tố, điều tra và xét xử.",
    5: "Cả 10 bài đều liên quan đến giải chạy bộ và marathon.",
    6: "Cả 10 bài đều liên quan đến bóng đá.",
    7: "Các bài liên quan đến bệnh lý, khám và điều trị.",
    8: "Cả 10 bài tập trung vào thị trường và biến động giá vàng.",
    9: "Các bài liên quan đến trường học và quản lý giáo dục.",
    10: "Các bài liên quan đến thời trang và phong cách người nổi tiếng.",
    11: "Các bài liên quan đến ca sĩ, âm nhạc và biểu diễn.",
    12: "Các bài liên quan đến AI, bảo mật và rủi ro công nghệ.",
    13: "Topic khá rộng; có một bài về quân sự lệch nhóm nội dung chính.",
    14: "Topic rộng về quy định, hành chính, thanh tra và quản lý dự án.",
}

keywords = pd.read_csv(
    ARTIFACT_DIR / "topic_keywords.csv",
    encoding="utf-8-sig",
)

documents = pd.read_csv(
    ARTIFACT_DIR / "document_topics.csv",
    encoding="utf-8-sig",
    dtype={"article_id": str},
)

required_keywords = {"topic_id", "rank", "keyword", "weight"}
required_documents = {
    "article_id",
    "split",
    "topic_id",
    "topic_probability",
    "title",
}

if not required_keywords.issubset(keywords.columns):
    raise ValueError("topic_keywords.csv thiếu cột bắt buộc")

if not required_documents.issubset(documents.columns):
    raise ValueError("document_topics.csv thiếu cột bắt buộc")

keywords["topic_id"] = pd.to_numeric(
    keywords["topic_id"], errors="raise"
).astype(int)

documents["topic_id"] = pd.to_numeric(
    documents["topic_id"], errors="raise"
).astype(int)

documents["topic_probability"] = pd.to_numeric(
    documents["topic_probability"], errors="coerce"
)

topic_ids = set(documents["topic_id"].unique()) - {-1}

if topic_ids != set(LABELS):
    raise ValueError("Danh sách topic không khớp 15 topic LDA Final")

development = documents.loc[
    documents["split"].isin(["train", "val"])
].copy()

development = development.sort_values(
    ["topic_id", "topic_probability", "article_id"],
    ascending=[True, False, True],
    kind="stable",
)

representatives = (
    development.groupby("topic_id", group_keys=False)
    .head(10)
    .copy()
)

representatives["rank"] = (
    representatives.groupby("topic_id").cumcount() + 1
)

representative_columns = [
    "topic_id",
    "rank",
    "article_id",
    "split",
    "title",
    "topic_probability",
    "second_topic_id",
    "second_topic_probability",
    "top_two_share_gap",
    "category_gold",
]

representatives = representatives[
    [col for col in representative_columns if col in representatives.columns]
]

representatives.to_csv(
    OUTPUT_DIR / "representative_docs_top10.csv",
    index=False,
    encoding="utf-8-sig",
)

keyword_summary = (
    keywords.sort_values(["topic_id", "rank"])
    .groupby("topic_id")["keyword"]
    .apply(lambda values: ", ".join(values.astype(str)))
    .reset_index(name="top_10_keywords")
)

counts = (
    documents.groupby("topic_id")
    .size()
    .reset_index(name="document_count")
)

review_path = OUTPUT_DIR / "topic_label_review.csv"
proposals_path = OUTPUT_DIR / "topic_label_proposals.csv"

if not review_path.exists():
    review = keyword_summary.merge(
        counts, on="topic_id", how="left"
    )
    review["proposed_label"] = ""
    review["reviewer"] = ""
    review["review_status"] = "pending"
    review["notes"] = ""

    review.to_csv(
        review_path,
        index=False,
        encoding="utf-8-sig",
    )

if proposals_path.exists():
    proposals = pd.read_csv(
        proposals_path,
        encoding="utf-8-sig",
    )
else:
    proposals = keyword_summary.merge(
        counts, on="topic_id", how="left"
    )
    proposals["reviewer"] = ""
    proposals["review_status"] = "proposed"
    proposals["notes"] = ""

proposals["topic_id"] = pd.to_numeric(
    proposals["topic_id"], errors="raise"
).astype(int)

if len(proposals) != 15 or set(proposals["topic_id"]) != set(LABELS):
    raise ValueError("File đề xuất phải chứa đúng 15 topic từ 0 đến 14")

proposals["proposed_label"] = proposals["topic_id"].map(LABELS)
proposals["notes"] = proposals["topic_id"].map(NOTES)

if "reviewer" not in proposals.columns:
    proposals["reviewer"] = ""

if "review_status" not in proposals.columns:
    proposals["review_status"] = "proposed"

if "notes" not in proposals.columns:
    proposals["notes"] = ""

proposals.to_csv(
    proposals_path,
    index=False,
    encoding="utf-8-sig",
)

check = pd.read_csv(
    proposals_path,
    encoding="utf-8-sig",
)

invalid = check["proposed_label"].astype(str).str.contains(
    "?", regex=False
).sum()

if invalid:
    raise RuntimeError(f"Còn {invalid} nhãn topic bị lỗi tiếng Việt")

for _, row in check.iterrows():
    topic_id = int(row["topic_id"])
    if row["proposed_label"] != LABELS[topic_id]:
        raise RuntimeError(f"Tên Topic {topic_id} không đúng UTF-8")

print("Đã cập nhật tên tiếng Việt cho 15 topic LDA.")
print("Số topic:", len(check))
print("Số nhãn lỗi:", invalid)
print("Tổng bài đại diện:", len(representatives))
print("File đề xuất:", proposals_path)

manifest_path = ARTIFACT_DIR / "manifest.json"
mapping_path = ARTIFACT_DIR / "topic_display_mapping.json"

with manifest_path.open("r", encoding="utf-8") as f:
    manifest = json.load(f)

if len(proposals) != 15:
    raise ValueError("Phải có đúng 15 topic LDA.")

if proposals["topic_id"].duplicated().any():
    raise ValueError("Có topic_id bị trùng.")

if set(proposals["topic_id"].astype(int)) != set(range(15)):
    raise ValueError("Topic ID phải từ 0 đến 14.")

mapping = {}

for _, row in proposals.iterrows():
    topic_id = int(row["topic_id"])
    label = str(row["proposed_label"]).strip()

    if not label or "?" in label:
        raise ValueError(f"Topic {topic_id} có nhãn không hợp lệ.")

    mapping[str(topic_id)] = {
        "suggest_label": label,
        "display_name": label,
        "display_group_id": None,
        "review_status": "proposed",
    }

payload = {
    "run_id": "lda_k15_seed42",
    "schema_version": manifest["schema_version"],
    "label_status": "proposed",
    "review_note": "TV3 proposed labels; independent review pending.",
    "mapping": mapping,
}

mapping_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Mapping:", mapping_path)
print("Topics:", len(mapping))
print("Status: proposed")
