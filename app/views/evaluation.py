"""Trang 4: Đánh giá và so sánh (so sánh mô hình, kết quả thí nghiệm, chuyên sâu)."""

from __future__ import annotations
import altair as alt
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from app.data_service import (
    ArtifactRun, get_available_splits, get_keywords_for_topic, get_model_overview,
    get_representatives_for_topic, get_topic_table, load_dashboard_bundle,
    load_projection,
)
from app.hdbscan_dashboard import render_hdbscan_details
from app.lda_dashboard import render_lda_details
from app.nmf_dashboard import render_nmf_details
from app.ui import charts
from app.ui.components import (
    build_topic_labels, data_table, label_of, note, page_header, table_download,
)
from app.ui.context import AnalysisContext
import json
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO_ROOT = Path(__file__).resolve().parents[2]

FIGURES = {
    "kmeans": [
        ("Chọn K", "results/kmeans/figures/kmeans_elbow_inertia.png"),
        ("Ablation P0–P3", "results/kmeans/figures/m03_ablation_metrics_comparison.png"),
        ("Title-only và Title+Content", "results/kmeans/figures/m03a_metrics_comparison.png"),
    ],
    "nmf": [
        ("Chọn số topic", "results/nmf/nmf_p3_select_review_seed42_v1/topic_count_selection.png"),
    ],
    "lda": [
        (
            "M08 — So sánh Coherence và Topic Diversity",
            "results/lda/figures/f01_m08_metrics_comparison.png",
        ),
        (
            "M08 — Thời gian chạy theo số topic K",
            "results/lda/figures/f02_m08_runtime_vs_k.png",
        ),
        (
            "M09 — Độ ổn định LDA giữa các cặp seed",
            "results/lda/figures/f03_m09_stability_pairs.png",
        ),
        (
            "M09 — Thời gian chạy theo random seed",
            "results/lda/figures/f04_m09_runtime_by_seed.png",
        ),
        (
            "LDA Final — Phân bố số lượng bài theo topic",
            "results/lda/figures/f05_final_topic_distribution.png",
        ),
        (
            "LDA Final — Topic Coherence NPMI",
            "results/lda/figures/f06_final_topic_npmi.png",
        ),
        (
            "LDA Final — Bản đồ PCA 2D",
            "results/lda/figures/f07_final_projection_2d.png",
        ),
    ],
    "hdbscan": [
        ("Số cụm", "results/hdbscan/figures/m11_cluster_count.png"),
        ("Outlier trên validation", "results/hdbscan/figures/m11_validation_noise_coverage.png"),
        ("ARI và NMI", "results/hdbscan/figures/m11_validation_ari_nmi.png"),
    ],
    "bertopic": [
        ("So sánh cấu hình", "results/bertopic_visualization/01_experiment_dashboard.png"),
        ("Title only và Title + Content", "results/bertopic_visualization/09_bertopic_title_content_ablation.png"),
        ("Từ khóa c-TF-IDF", "results/bertopic_visualization/06_e1_ctfidf_keywords.png"),
        ("Đánh giá cụm", "results/bertopic_visualization/08_e1_cluster_evaluation.png"),
    ],
}

TABLES = {
    "kmeans": [
        "results/kmeans/kmeans_k_search_metrics.csv",
        "results/kmeans/m03_preprocessing_ablation_metrics.csv",
        "results/kmeans/m03a_title_vs_content_metrics.csv",
    ],
    "nmf": [
        "results/nmf/nmf_p3_select_review_seed42_v1/topic_count_metrics.csv",
        "results/nmf/nmf_p3_k10_seed42_dashboard_v2/split_metrics.csv",
        "results/nmf/nmf_p3_k10_seed42_dashboard_v2/runtime_by_stage.csv",
        "results/nmf/nmf_p3_k10_seed42_dashboard_v2/multi_topic_cases.csv",
    ],
    "lda": [
        "results/lda/m08/topic_number_results.csv",
        "results/lda/m09/stability_metrics.csv",
        "results/lda/m09/runtime_by_seed.csv",
    ],
    "hdbscan": [
        "results/hdbscan/m11_summary.csv",
        "results/hdbscan/validation/m11_validation_metrics.csv",
        "results/hdbscan/outlier_analysis/m12_outlier_rates.csv",
        "results/hdbscan/stability/minilm_title_content_eom_15_5_seeds42_51/stability_summary.csv",
        "results/hdbscan/stability/minilm_title_content_eom_15_5_seeds42_51/pairwise_keyword_jaccard.csv",
        "results/hdbscan/membership/minilm_title_content_eom_15_5/membership_summary.csv",
    ],
    "bertopic": ["results/bertopic_experiments/summary.csv"],
}

BERTOPIC_SUMMARY_COLUMNS = {
    "run_id": "Run", "input_text_label": "Đầu vào", "topic_count": "Số topic",
    "train_outlier_rate": "Outlier train", "validation_outlier_rate": "Outlier validation",
    "topic_diversity": "Diversity", "validation_adjusted_rand_index": "ARI validation",
    "validation_normalized_mutual_info": "NMI validation",
    "validation_npmi_coherence": "NPMI validation", "test_outlier_rate": "Outlier test",
    "test_adjusted_rand_index": "ARI test", "test_normalized_mutual_info": "NMI test",
    "test_npmi_coherence": "NPMI test",
}
BERTOPIC_TIMING_COLUMNS = {
    "run_id": "Run", "model_load_seconds": "Load model", "representation_seconds": "Embedding",
    "model_init_seconds": "Khởi tạo", "fit_seconds": "Fit",
    "validation_predict_seconds": "Predict validation", "test_predict_seconds": "Predict test",
    "evaluation_seconds": "Evaluation", "runtime_seconds": "Runtime pipeline",
}


def render(ctx: AnalysisContext) -> None:
    page_header(
        "Đánh giá và so sánh",
        "Đối chiếu các phương pháp, xem kết quả thí nghiệm và phân tích chuyên sâu theo mô hình.",
        {"Run đang chọn": ctx.run.label},
        eyebrow="Đánh giá",
    )
    compare_tab, experiment_tab, deep_tab = st.tabs(
        ["So sánh mô hình", "Kết quả thí nghiệm", "Chuyên sâu theo phương pháp"]
    )
    with compare_tab:
        _render_comparison(ctx.runs, ctx.run)
    with experiment_tab:
        _render_experiments(ctx.run)
    with deep_tab:
        _render_deep_dive(ctx)


def comparison_table(runs: list[ArtifactRun], split: str) -> pd.DataFrame:
    rows = []
    for run in runs:
        bundle = load_dashboard_bundle(str(run.path))
        available = get_available_splits(bundle)
        if split != "all" and split not in available:
            continue
        overview = get_model_overview(bundle, split, artifact_dir=run.path)
        ari = overview["adjusted_rand_index"]
        nmi = overview["normalized_mutual_info"]
        coherence = overview["coherence"]

        if bundle.model_summary.algorithm.strip().lower() == "hdbscan":
            metric_docs = bundle.document_topics
            if split != "all":
                metric_docs = metric_docs.loc[
                    metric_docs["split"].eq(split)
                ]

            if (
                not metric_docs.empty
                and metric_docs["category_gold"].notna().all()
                and metric_docs["topic_id"].notna().all()
            ):
                ari = round(float(adjusted_rand_score(
                    metric_docs["category_gold"],
                    metric_docs["topic_id"],
                )), 4)
                nmi = round(float(normalized_mutual_info_score(
                    metric_docs["category_gold"],
                    metric_docs["topic_id"],
                )), 4)

            coherence_path = run.path / "coherence_summary.json"
            if coherence_path.is_file():
                coherence_data = json.loads(
                    coherence_path.read_text(encoding="utf-8-sig")
                )
                if coherence_data.get("run_id") != bundle.model_summary.run_id:
                    raise ValueError("Coherence artifact không đúng run_id")

                split_result = coherence_data.get(
                    "scores_by_split", {}
                ).get(split, {})
                coherence = split_result.get("npmi_coherence")
        rows.append({
            "Mô hình": bundle.model_summary.algorithm,
            "Run": bundle.model_summary.run_id,
            "Split": split if split != "all" else ", ".join(available),
            "Số bài": overview["document_count"],
            "Số topic": overview["topic_count"],
            "Outlier áp dụng": "Có" if overview["supports_outlier"] else "Không",
            "Outlier (%)": (
                round(100 * overview["outlier_rate"], 2) if overview["outlier_rate"] is not None else None
            ),
            "Topic diversity (run)": overview["topic_diversity"],
            "ARI (run)": ari,
            "NMI (run)": nmi,
            "Coherence (run)": coherence,
            "Fit (giây, run)": overview["fit_seconds"],
            "Runtime pipeline (giây, run)": overview["runtime_seconds"],
            "Biểu diễn": overview["representation"],
        })
    return pd.DataFrame(rows)


def _render_comparison(runs: list[ArtifactRun], current: ArtifactRun) -> None:
    note(
        "Trang này so sánh nhiều run nên không phụ thuộc split ở sidebar. Chọn cùng một split để "
        "đối chiếu số bài và số topic; metric run giữ đúng phạm vi mà nhóm tác giả đã báo cáo."
    )
    split = st.selectbox("Split dùng để so sánh", ["train", "val", "test", "all"], key="compare_split")
    table = comparison_table(runs, split)
    if table.empty:
        st.info("Không có run nào xuất split đã chọn.")
        return
    data_table(table)
    table_download(table, f"model_comparison_{split}.csv", "Tải bảng so sánh CSV")
    if split == "all":
        note(
            "Mỗi mô hình có phạm vi split khác nhau; hàng 'all' dùng để khám phá, "
            "không dùng làm kết luận so sánh định lượng.", "warn",
        )
    note(
        "ARI/NMI dùng category_gold rộng; coherence, topic diversity, runtime và outlier có định nghĩa/phạm vi "
        "khác nhau giữa các phương pháp. Ô N/A nghĩa là run chưa xuất metric, không phải giá trị bằng 0."
    )
    st.subheader("Đối sánh chỉ số")
    st.plotly_chart(
        charts.model_compare_bar(table, ["ARI (run)", "NMI (run)", "Topic diversity (run)"]),
        width="stretch", config=charts.PLOT_CONFIG,
    )
    st.subheader("Xem hai run cạnh nhau")
    options = [r for r in runs if split == "all" or split in get_available_splits(load_dashboard_bundle(str(r.path)))]
    if len(options) < 2:
        return
    first = st.selectbox(
        "Run A", options, format_func=lambda r: r.label,
        index=options.index(current) if current in options else 0, key="compare_run_a",
    )
    second = st.selectbox(
        "Run B", options, format_func=lambda r: r.label, index=1 if len(options) > 1 else 0, key="compare_run_b",
    )
    for column, run in zip(st.columns(2), (first, second)):
        bundle = load_dashboard_bundle(str(run.path))
        with column:
            st.markdown(f"**{run.label}**")
            data_table(get_topic_table(bundle, split=split).head(8)[["topic_id", "topic_name", "document_count"]])
            with st.expander("Cấu hình run"):
                st.json(bundle.model_summary.config)

def _render_hdbscan_stability_charts(frame: pd.DataFrame) -> None:
    required = {
        "seed", "topic_count", "outlier_rate",
        "coverage", "ari_between_runs", "keyword_jaccard",
    }
    if not required.issubset(frame.columns):
        st.warning("Stability Summary thiếu cột để vẽ biểu đồ.")
        return

    data = frame[list(required)].copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["seed"]).sort_values("seed")
    data["seed"] = data["seed"].astype(int)

    st.markdown("**Số cụm theo seed UMAP**")
    count_chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("seed:O", title="Seed UMAP"),
            y=alt.Y("topic_count:Q", title="Số cụm"),
            tooltip=["seed:O", "topic_count:Q"],
        )
        .properties(height=260)
    )
    st.altair_chart(count_chart, use_container_width=True)

    rates = data[
        ["seed", "outlier_rate", "coverage"]
    ].melt(
        id_vars="seed",
        var_name="metric",
        value_name="value",
    )
    rates["metric"] = rates["metric"].map({
        "outlier_rate": "Tỷ lệ nhiễu",
        "coverage": "Độ bao phủ",
    })

    st.markdown("**Tỷ lệ nhiễu và độ bao phủ**")
    rate_chart = (
        alt.Chart(rates)
        .mark_line(point=True)
        .encode(
            x=alt.X("seed:O", title="Seed UMAP"),
            y=alt.Y(
                "value:Q",
                title="Tỷ lệ",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("metric:N", title="Chỉ số"),
            tooltip=[
                "seed:O",
                "metric:N",
                alt.Tooltip("value:Q", format=".2%"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(rate_chart, use_container_width=True)

    scores = data[
        ["seed", "ari_between_runs", "keyword_jaccard"]
    ].melt(
        id_vars="seed",
        var_name="metric",
        value_name="value",
    )
    scores["metric"] = scores["metric"].map({
        "ari_between_runs": "ARI giữa các run",
        "keyword_jaccard": "Keyword Jaccard",
    })

    st.markdown("**Độ nhất quán giữa các seed**")
    score_chart = (
        alt.Chart(scores)
        .mark_line(point=True)
        .encode(
            x=alt.X("seed:O", title="Seed UMAP"),
            y=alt.Y(
                "value:Q",
                title="Điểm",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("metric:N", title="Chỉ số"),
            tooltip=[
                "seed:O",
                "metric:N",
                alt.Tooltip("value:Q", format=".3f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(score_chart, use_container_width=True)

    st.caption(
        "ARI và Keyword Jaccard tại mỗi seed là giá trị trung bình "
        "của các cặp chứa seed đó; các cặp không phải thí nghiệm độc lập."
    )

def _render_hdbscan_m12_charts(frame: pd.DataFrame) -> None:
    required = {
        "embedding_run", "experiment", "group_type", "group_value",
        "n_articles", "n_outliers", "outlier_percent",
    }
    if not required.issubset(frame.columns):
        st.warning("M12 Outlier Rates thiếu cột để vẽ biểu đồ.")
        return

    data = frame.copy()
    for column in ("n_articles", "n_outliers", "outlier_percent"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    embeddings = sorted(data["embedding_run"].dropna().unique())
    default_embedding = "minilm_title_content"
    embedding = st.selectbox(
        "Embedding",
        embeddings,
        index=(
            embeddings.index(default_embedding)
            if default_embedding in embeddings else 0
        ),
        key="m12_chart_embedding",
    )

    available = data.loc[data["embedding_run"].eq(embedding)]
    experiments = sorted(available["experiment"].dropna().unique())
    default_experiment = "eom_15_5"
    experiment = st.selectbox(
        "Cấu hình HDBSCAN",
        experiments,
        index=(
            experiments.index(default_experiment)
            if default_experiment in experiments else 0
        ),
        key="m12_chart_experiment",
    )

    selected = available.loc[
        available["experiment"].eq(experiment)
    ].copy()

    overall = selected.loc[selected["group_type"].eq("overall")]
    if not overall.empty:
        row = overall.iloc[0]
        st.metric(
            "Tỷ lệ outlier chung",
            f"{row['outlier_percent']:.2f}%",
            help=(
                f"{int(row['n_outliers']):,} bài outlier trên "
                f"{int(row['n_articles']):,} bài train"
            ),
        )

    for group_type, title, order in (
        (
            "content_length",
            "Theo độ dài nội dung",
            ["0-99", "100-299", "300-599", "600+"],
        ),
        (
            "source",
            "Theo nguồn báo",
            ["tuoitre", "vietnamnet", "vnexpress"],
        ),
    ):
        subset = selected.loc[
            selected["group_type"].eq(group_type)
        ].dropna(subset=["outlier_percent"])

        if subset.empty:
            continue

        st.markdown(f"**{title}**")
        chart = (
            alt.Chart(subset)
            .mark_bar()
            .encode(
                y=alt.Y(
                    "group_value:N",
                    title="Nhóm",
                    sort=order,
                ),
                x=alt.X(
                    "outlier_percent:Q",
                    title="Tỷ lệ outlier (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                tooltip=[
                    alt.Tooltip("group_value:N", title="Nhóm"),
                    alt.Tooltip(
                        "n_articles:Q",
                        title="Số bài",
                        format=",.0f",
                    ),
                    alt.Tooltip(
                        "n_outliers:Q",
                        title="Số outlier",
                        format=",.0f",
                    ),
                    alt.Tooltip(
                        "outlier_percent:Q",
                        title="Outlier (%)",
                        format=".2f",
                    ),
                ],
            )
            .properties(height=max(180, len(subset) * 45))
        )
        st.altair_chart(chart, use_container_width=True)

    st.caption(
        "Tỷ lệ được tính trên tập train cho cấu hình đã chọn. "
        "Các nhóm có số bài khác nhau; xem số bài trong tooltip. "
        "Cột outlier_percent của CSV đã ở đơn vị phần trăm."
    )

def _render_experiments(run: ArtifactRun) -> None:
    note("Hình và bảng kết quả đã được nhóm xuất trong repo. Các phép đo có protocol riêng cho từng phương pháp.")
    for title, relative_path in FIGURES.get(run.method, []):
        path = REPO_ROOT / relative_path
        if path.is_file():
            st.subheader(title)
            st.image(str(path), width="stretch")
    for relative_path in TABLES.get(run.method, []):
        path = REPO_ROOT / relative_path
        if not path.is_file():
            continue
        with st.expander(path.stem.replace("_", " ").title()):
            frame = pd.read_csv(path, encoding="utf-8-sig")
            if run.method == "bertopic" and path.name == "summary.csv":
                _render_bertopic_summary(frame)
            elif (
                run.method == "hdbscan"
                and path.name == "stability_summary.csv"
            ):
                _render_hdbscan_stability_charts(frame)
                if st.checkbox(
                    "Hiện bảng số liệu gốc",
                    value=False,
                    key="hdbscan_stability_raw_table",
                ):
                    data_table(frame.head(100))
            elif (
                run.method == "hdbscan"
                and path.name == "m12_outlier_rates.csv"
            ):
                _render_hdbscan_m12_charts(frame)
                if st.checkbox(
                    "Hiện bảng M12 gốc",
                    value=False,
                    key="hdbscan_m12_raw_table",
                ):
                    data_table(frame.head(100))
                    st.caption(f"Hiển thị tối đa 100/{len(frame):,} hàng.")
            else:
                data_table(frame.head(100))
            table_download(frame, path.name, "Tải bảng CSV")
    if run.method == "bertopic":
        _render_bertopic_config(run)
    if not FIGURES.get(run.method) and not TABLES.get(run.method):
        st.info("Chưa có hình hoặc bảng thí nghiệm được liên kết cho run này.")


def _render_bertopic_summary(frame: pd.DataFrame) -> None:
    compact = frame[[c for c in BERTOPIC_SUMMARY_COLUMNS if c in frame]].rename(columns=BERTOPIC_SUMMARY_COLUMNS)
    for column in ("Outlier train", "Outlier validation", "Outlier test"):
        if column in compact:
            compact[column] = 100 * compact[column]
    st.markdown("**Chất lượng và khả năng bao phủ**")
    percent = st.column_config.NumberColumn(format="%.1f%%")
    score = st.column_config.NumberColumn(format="%.3f")
    data_table(compact, {
        "Outlier train": percent, "Outlier validation": percent, "Outlier test": percent,
        "Diversity": score, "ARI validation": score, "NMI validation": score,
        "NPMI validation": score, "ARI test": score, "NMI test": score, "NPMI test": score,
    })
    timings = frame[[c for c in BERTOPIC_TIMING_COLUMNS if c in frame]].rename(columns=BERTOPIC_TIMING_COLUMNS)
    st.markdown("**Thời gian theo giai đoạn (giây)**")
    data_table(timings)
    st.caption("N/A nghĩa là run cũ chưa được đo lại theo protocol mới. Không được diễn giải N/A là 0 giây.")


def _render_bertopic_config(
    run: ArtifactRun,
    *,
    show_topic_keywords: bool = True,
) -> None:
    config_path = REPO_ROOT / "configs/bertopic_experiments.yaml"
    if not config_path.is_file():
        return
    with config_path.open("r", encoding="utf-8") as file:
        experiment_config = yaml.safe_load(file)
    rows = [
        {
            "Run": item["run_id"],
            "Đầu vào": item.get("input_text_label", "title_content"),
            "Embedding": item["embedding"]["model_name"],
            "UMAP neighbors": item["umap"]["n_neighbors"],
            "UMAP components": item["umap"]["n_components"],
            "UMAP min_dist": item["umap"]["min_dist"],
            "UMAP metric": item["umap"]["metric"],
            "HDBSCAN min cluster": item["hdbscan"]["min_cluster_size"],
            "HDBSCAN min samples": item["hdbscan"]["min_samples"],
            "Selection": item["hdbscan"]["cluster_selection_method"],
        }
        for item in experiment_config["runs"]
    ]
    st.subheader("Cấu hình UMAP và HDBSCAN")
    data_table(pd.DataFrame(rows))
    note(
        "E2 chỉ đổi UMAP n_neighbors; E3 chỉ đổi HDBSCAN min_cluster_size; E4 chỉ đổi đầu vào "
        "sang title-only. E1 được chọn bằng validation trước khi mở test."
    )
    if not show_topic_keywords:
        return
    bundle = load_dashboard_bundle(str(run.path))
    topics = get_topic_table(bundle).head(15)
    if topics.empty:
        return
    labels = build_topic_labels(bundle, run.run_id)
    topic_id = st.selectbox(
        "Xem từ khóa c-TF-IDF của chủ đề", topics["topic_id"].astype(int).tolist(),
        format_func=lambda v: label_of(v, labels), key="experiment_ctfidf_topic",
    )
    keywords = get_keywords_for_topic(bundle, topic_id).head(15)
    if keywords.empty:
        return
    score_column = "weight" if "weight" in keywords.columns else "score"
    keywords[score_column] = pd.to_numeric(keywords[score_column], errors="coerce")
    st.plotly_chart(
        charts.keyword_bar(keywords, score_column, "Điểm c-TF-IDF"),
        width="stretch", config=charts.PLOT_CONFIG,
    )
    data_table(keywords)
    st.caption("c-TF-IDF cao nghĩa là từ đặc trưng cho topic so với các topic khác; không phải xác suất xuất hiện của từ.")


def _render_deep_dive(ctx: AnalysisContext) -> None:
    run = ctx.run
    if run.method == "bertopic":
        _render_bertopic_deep_dive(ctx)
    elif run.method == "nmf":
        st.subheader("Phân tích chi tiết NMF")
        render_nmf_details(ctx.bundle, str(run.path))

    elif run.method == "hdbscan":
        st.subheader("Phân tích chi tiết HDBSCAN")
        render_hdbscan_details(ctx.bundle, str(run.path))

    elif run.method == "lda":
        render_lda_details(ctx.bundle, str(run.path))

    else:
        st.info(
            "Phân tích chuyên sâu hiện có cho BERTopic, NMF, HDBSCAN và LDA. "
            "Chọn một run tương ứng ở sidebar để xem."
        )


def _render_bertopic_deep_dive(ctx: AnalysisContext) -> None:
    """Phân tích tương tác BERTopic từ artifact, không phụ thuộc ảnh tĩnh."""
    run, bundle, split = ctx.run, ctx.bundle, ctx.split
    st.subheader("Phân tích chi tiết BERTopic")
    note(
        "BERTopic là pipeline gồm biểu diễn ngữ nghĩa, UMAP, HDBSCAN và c-TF-IDF. "
        "Các biểu đồ dưới đây tách riêng chất lượng, chi phí chạy và nội dung topic."
    )

    summary_path = REPO_ROOT / "results/bertopic_experiments/summary.csv"
    if summary_path.is_file():
        summary = pd.read_csv(summary_path, encoding="utf-8-sig")
        summary = summary.loc[summary["run_id"].ne("smoke_200")].copy()
        order = {
            "e0_multilingual_minilm": 0,
            "e1_vietnamese_sbert": 1,
            "e2_vietnamese_sbert_neighbors30": 2,
            "e3_vietnamese_sbert_cluster30": 3,
            "e4_vietnamese_sbert_title_only": 4,
        }
        summary["_order"] = summary["run_id"].map(order).fillna(99)
        summary = summary.sort_values("_order").drop(columns="_order")

        st.markdown("#### So sánh E0–E4")
        quality = pd.DataFrame({
            "Mô hình": "BERTopic",
            "Run": summary["run_id"],
            "Outlier validation (%)": 100 * summary["validation_outlier_rate"],
            "Topic diversity (%)": 100 * summary["topic_diversity"],
            "NMI validation (%)": 100 * summary["validation_normalized_mutual_info"],
            "ARI validation (%)": 100 * summary["validation_adjusted_rand_index"],
        })
        st.plotly_chart(
            charts.model_compare_bar(
                quality,
                [
                    "Outlier validation (%)", "Topic diversity (%)",
                    "NMI validation (%)", "ARI validation (%)",
                ],
            ),
            width="stretch",
            config=charts.PLOT_CONFIG,
        )
        note(
            "Outlier thấp là tốt, trong khi diversity, NMI và ARI cao là tốt. "
            "Không chọn cấu hình chỉ dựa trên một cột duy nhất.",
            "warn",
        )

        timing = pd.DataFrame({
            "Mô hình": "BERTopic",
            "Run": summary["run_id"],
            "Load model": summary.get("model_load_seconds"),
            "Embedding": summary.get("representation_seconds"),
            "Fit": summary.get("fit_seconds"),
            "Predict validation": summary.get("validation_predict_seconds"),
            "Predict test": summary.get("test_predict_seconds"),
        })
        st.markdown("#### Runtime theo giai đoạn")
        st.plotly_chart(
            charts.model_compare_bar(
                timing,
                ["Load model", "Embedding", "Fit", "Predict validation", "Predict test"],
            ),
            width="stretch",
            config=charts.PLOT_CONFIG,
        )
        st.caption(
            "Đơn vị là giây. N/A nghĩa là run chưa được đo ở giai đoạn đó; "
            "không được xem là 0 giây."
        )

    _render_bertopic_config(run, show_topic_keywords=False)

    topics = get_topic_table(bundle, split=split).head(30)
    if topics.empty:
        st.info("Không có topic BERTopic trong split đang chọn.")
        return
    labels = build_topic_labels(bundle, run.run_id)
    topic_ids = topics["topic_id"].astype(int).tolist()
    topic_id = st.selectbox(
        "Topic cần phân tích",
        topic_ids,
        format_func=lambda value: label_of(value, labels),
        key="bertopic_deep_topic",
    )

    projection = load_projection(str(run.path))
    if not projection.empty:
        if split != "all" and "split" in projection:
            projection = projection.loc[projection["split"].eq(split)]
        visible_topics = topic_ids[:8]
        if topic_id not in visible_topics:
            visible_topics.append(topic_id)
        show_outliers = st.checkbox(
            "Hiển thị outlier trên bản đồ",
            value=False,
            key="bertopic_deep_outliers",
        )
        st.markdown("#### Bản đồ biểu diễn 2D")
        st.plotly_chart(
            charts.projection_scatter(
                projection,
                labels,
                visible_topics,
                selected=topic_id,
                show_outliers=show_outliers,
            ),
            width="stretch",
            config=charts.PLOT_CONFIG,
        )
        st.caption(
            "UMAP 2D chỉ hỗ trợ quan sát cấu trúc tương đối; không dùng riêng biểu đồ này "
            "để kết luận mô hình tốt hay xấu."
        )

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Từ khóa c-TF-IDF")
        keywords = get_keywords_for_topic(bundle, topic_id).head(15)
        if keywords.empty:
            st.info("Topic này chưa xuất từ khóa.")
        else:
            score_column = "weight" if "weight" in keywords.columns else "score"
            keywords[score_column] = pd.to_numeric(keywords[score_column], errors="coerce")
            st.plotly_chart(
                charts.keyword_bar(keywords, score_column, "Điểm c-TF-IDF"),
                width="stretch",
                config=charts.PLOT_CONFIG,
            )
            data_table(keywords)
    with right:
        st.markdown("#### Bài báo đại diện")
        representatives = get_representatives_for_topic(
            str(run.path), topic_id, split=split,
        )
        if representatives.empty:
            st.info("Topic này chưa xuất bài báo đại diện cho split đang chọn.")
        else:
            columns = [
                column for column in (
                    "rank", "title", "split", "topic_probability", "category_gold"
                )
                if column in representatives
            ]
            data_table(representatives[columns].head(10))
    note(
        "c-TF-IDF mô tả từ đặc trưng của cả topic, còn bài đại diện giúp người đánh giá "
        "kiểm tra tên topic có đúng với nội dung thực tế hay không."
    )
