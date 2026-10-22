"""Kiểm thử dịch vụ bóc tách, phân loại chủ đề và gom nhóm sự kiện bài báo mới."""

from __future__ import annotations

import json

from app.article_organizer import (
    SAMPLE_ARTICLES,
    assign_topics_to_articles,
    cluster_articles_into_events,
    extract_article_from_html,
    find_related_corpus_articles,
    get_csv_template_bytes,
    parse_raw_text_entries,
    parse_uploaded_file,
    preprocess_vietnamese_texts,
)
from app.data_service import discover_artifact_runs, load_dashboard_bundle
from app.ui.components import build_topic_labels


def test_extract_article_from_html_vnexpress_and_generic() -> None:
    html_vne = """
    <html>
      <head>
        <meta property="article:published_time" content="2026-09-15T10:30:00+07:00" />
      </head>
      <body>
        <h1 class="title-detail">Ngân hàng giảm lãi suất cho vay mua nhà</h1>
        <p class="description">Nhiều ngân hàng thương mại tung gói tín dụng ưu đãi.</p>
        <article class="fck_detail">
          <p>Lãi suất vay mua nhà giảm mạnh trong quý 3 nhằm kích cầu tín dụng.</p>
          <p>Doanh nghiệp bất động sản kỳ vọng thanh khoản sớm phục hồi.</p>
        </article>
      </body>
    </html>
    """
    parsed = extract_article_from_html(html_vne, url="https://vnexpress.net/ngan-hang-giam-lai-suat-123.html")
    assert parsed["source"] == "VnExpress"
    assert "Ngân hàng giảm lãi suất" in parsed["title"]
    assert "tín dụng" in parsed["content"]
    assert parsed["published_at"].startswith("2026-09-15")


def test_parse_raw_text_and_uploaded_files() -> None:
    entries = parse_raw_text_entries(
        title="Tiêu đề đơn",
        content="Nội dung bài viết đơn về kinh tế vĩ mô.",
        batch_text="Bài số hai\nNội dung bài báo thứ hai về bóng đá Việt Nam.\n---\nBài số ba\nNội dung bài báo thứ ba về tuyển sinh đại học.",
    )
    assert len(entries) == 3
    assert entries[0]["title"] == "Tiêu đề đơn"
    assert entries[1]["title"] == "Bài số hai"
    assert entries[2]["title"] == "Bài số ba"

    # Kiểm thử đọc CSV template
    csv_bytes = get_csv_template_bytes()
    csv_articles = parse_uploaded_file("template.csv", csv_bytes)
    assert len(csv_articles) == 3
    assert all(item["title"] and item["content"] for item in csv_articles)

    # Kiểm thử đọc JSON
    json_bytes = json.dumps(SAMPLE_ARTICLES[:2], ensure_ascii=False).encode("utf-8")
    json_articles = parse_uploaded_file("sample.json", json_bytes)
    assert len(json_articles) == 2


def test_preprocess_and_event_clustering() -> None:
    processed = preprocess_vietnamese_texts([a["content"] for a in SAMPLE_ARTICLES])
    assert len(processed) == len(SAMPLE_ARTICLES)
    assert any("_" in text for text in processed)

    organized_df, event_summary = cluster_articles_into_events(
        SAMPLE_ARTICLES,
        similarity_threshold=0.15,
    )
    assert len(organized_df) == len(SAMPLE_ARTICLES)
    assert not event_summary.empty
    # Hai bài ngân hàng / lãi suất (index 0 và 1) phải nằm cùng một nhóm sự kiện
    assert int(organized_df.iloc[0]["event_id"]) == int(organized_df.iloc[1]["event_id"])
    # Hai bài bóng đá (index 2 và 3) phải nằm cùng một nhóm sự kiện
    assert int(organized_df.iloc[2]["event_id"]) == int(organized_df.iloc[3]["event_id"])
    # Hai bài giáo dục / tuyển sinh (index 4 và 5) phải nằm cùng một nhóm sự kiện
    assert int(organized_df.iloc[4]["event_id"]) == int(organized_df.iloc[5]["event_id"])


def test_assign_topics_across_all_five_models() -> None:
    runs = {run.method: run for run in discover_artifact_runs()}
    for method in ("bertopic", "hdbscan", "kmeans", "lda", "nmf"):
        run = runs[method]
        bundle = load_dashboard_bundle(str(run.path))
        labels = build_topic_labels(bundle, run.run_id)
        assigned = assign_topics_to_articles(SAMPLE_ARTICLES[:4], bundle, run, labels)
        assert len(assigned) == 4
        assert "topic_id" in assigned.columns
        assert "topic_label" in assigned.columns
        assert "confidence" in assigned.columns
        assert assigned["confidence"].between(0.0, 1.0).all()

    # Kiểm thử tìm bài báo liên quan trong kho 9,274 bài gốc
    nmf_run = runs["nmf"]
    nmf_bundle = load_dashboard_bundle(str(nmf_run.path))
    nmf_labels = build_topic_labels(nmf_bundle, nmf_run.run_id)
    processed = preprocess_vietnamese_texts([SAMPLE_ARTICLES[0]["content"]])
    related = find_related_corpus_articles(processed, nmf_bundle, nmf_labels, top_k=4)
    assert not related.empty
    assert len(related) <= 4
    assert (related["similarity"] > 0).all()

