"""Kiểm thử lớp dữ liệu dành cho Streamlit."""

from app.data_service import (
    discover_artifact_runs,
    enrich_documents,
    get_available_splits,
    get_document_table,
    get_documents_for_topic,
    get_keywords_for_topic,
    get_model_overview,
    get_representatives_for_topic,
    get_topic_table,
    load_dashboard_bundle,
    load_projection,
)


ARTIFACT_DIR = (
    "artifacts/common/bertopic/"
    "e1_vietnamese_sbert"
)


def test_load_dashboard_bundle() -> None:
    """Streamlit đọc được artifact BERTopic thật."""
    bundle = load_dashboard_bundle(ARTIFACT_DIR)

    assert bundle.model_summary.run_id == (
        "e1_vietnamese_sbert"
    )
    assert len(bundle.document_topics) == 7884
    assert len(bundle.topic_keywords) == 960


def test_get_model_overview() -> None:
    """Dashboard có đủ số liệu tổng quan."""
    bundle = load_dashboard_bundle(ARTIFACT_DIR)

    overview = get_model_overview(bundle)

    assert overview["algorithm"] == "BERTopic"
    assert overview["topic_count"] == 96
    assert overview["document_count"] == 7884
    assert (
        overview["assigned_document_count"]
        + overview["outlier_document_count"]
        == overview["document_count"]
    )


def test_get_topic_table_excludes_outlier() -> None:
    """Bảng topic mặc định không hiển thị topic -1."""
    bundle = load_dashboard_bundle(ARTIFACT_DIR)

    topics = get_topic_table(bundle)

    assert -1 not in topics["topic_id"].tolist()
    assert topics["document_count"].is_monotonic_decreasing


def test_get_topic_details() -> None:
    """Có thể lấy bài báo và từ khóa theo topic."""
    bundle = load_dashboard_bundle(ARTIFACT_DIR)

    topic_id = int(
        get_topic_table(bundle).iloc[0]["topic_id"]
    )

    documents = get_documents_for_topic(
        bundle,
        topic_id,
    )
    keywords = get_keywords_for_topic(
        bundle,
        topic_id,
    )

    assert not documents.empty
    assert not keywords.empty
    assert documents["topic_id"].eq(topic_id).all()
    assert keywords["topic_id"].eq(topic_id).all()


def test_five_models_are_discoverable_and_loadable() -> None:
    """Bộ chọn mô hình chỉ liệt kê manifest có thể đọc."""
    runs = discover_artifact_runs()

    assert {run.method for run in runs} == {
        "bertopic", "hdbscan", "kmeans", "lda", "nmf"
    }
    for run in runs:
        bundle = load_dashboard_bundle(str(run.path))
        assert bundle.manifest.run_id == run.run_id
        assert not bundle.document_topics.empty


def test_split_counts_are_derived_from_assignments() -> None:
    """Số bài trên UI phải khớp split, kể cả khi summary khác scope."""
    bundle = load_dashboard_bundle(ARTIFACT_DIR)
    assert get_available_splits(bundle) == ["train", "val"]

    overview = get_model_overview(bundle, "val")
    topics = get_topic_table(bundle, split="val")
    documents = get_document_table(bundle, "val")

    assert overview["document_count"] == 1391
    assert topics["document_count"].sum() == overview["assigned_document_count"]
    assert len(documents) == 1391


def test_optional_map_and_metadata() -> None:
    """Các run đã bàn giao projection và metadata tìm kiếm tải được."""
    runs = {run.method: run for run in discover_artifact_runs()}
    assert not load_projection(str(runs["lda"].path)).empty
    assert not load_projection(str(runs["kmeans"].path)).empty

    bundle = load_dashboard_bundle(str(runs["kmeans"].path))
    documents = enrich_documents(get_document_table(bundle, "train"))
    assert documents["source"].notna().any()
    assert documents["published_at"].notna().any()


def test_hdbscan_validation_artifacts_integrate_with_ui() -> None:
    """Artifact HDBSCAN mới phải cấp đủ validation, map và bài đại diện."""
    runs = {run.method: run for run in discover_artifact_runs()}
    run = runs["hdbscan"]
    bundle = load_dashboard_bundle(str(run.path))

    assert get_available_splits(bundle) == ["train", "val", "test"]
    assert len(get_document_table(bundle, "train")) == 6493
    assert len(get_document_table(bundle, "val")) == 1391
    assert len(get_document_table(bundle, "test")) == 1390

    projection = load_projection(str(run.path))
    assert len(projection) == 9274
    assert set(projection["split"]) == {"train", "val", "test"}
    assert projection["article_id"].is_unique

    representatives = get_representatives_for_topic(str(run.path), 1, "val")
    assert not representatives.empty
    assert representatives["topic_id"].eq(1).all()
    assert representatives["split"].eq("val").all()
