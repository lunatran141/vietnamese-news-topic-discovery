"""Đo lại pipeline K đã khóa và xuất dữ liệu giao diện, giữ nguyên run cũ."""

from __future__ import annotations

import argparse
import os
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiments.nmf.run_experiment import (
    base_metadata, build_document_outputs, build_model, build_vectorizer,
    create_run_directories, evaluate_model, load_dataset_v1, load_yaml,
    resolve_project_path, write_json,
)


def prepare_dashboard(config: dict, dataset: pd.DataFrame, run_id: str) -> tuple:
    """Validation fit trên train; final fit trên development, không chọn lại K."""
    k = int(config["topic_selection"]["selected_n_topics"])
    if k not in config["topic_selection"]["candidates"]:
        raise ValueError("K phải là cấu hình đã được khảo sát trên validation")
    run_id, artifact_dir, result_dir = create_run_directories(config, "dashboard", run_id)
    phases = {name: 0.0 for name in (
        "representation_seconds", "fit_seconds", "validation_transform_seconds",
        "test_transform_seconds", "evaluation_seconds",
    )}
    timing_rows, metric_rows = [], []
    text = config["data"]["text_column"]
    gold = config["data"]["gold_column"]
    train = dataset.loc[dataset["split"].eq("train")].reset_index(drop=True)
    val = dataset.loc[dataset["split"].eq("val")].reset_index(drop=True)
    development = dataset.loc[dataset["split"].isin(["train", "val"])].reset_index(drop=True)
    test = dataset.loc[dataset["split"].eq("test")].reset_index(drop=True)

    for stage, fit_data, evaluation_data in (
        ("validation", train, val), ("test", development, test),
    ):
        stage_times = dict.fromkeys(phases, 0.0)
        start = perf_counter()
        vectorizer = build_vectorizer(config)
        matrix = vectorizer.fit_transform(fit_data[text].astype(str))
        features = vectorizer.get_feature_names_out()
        stage_times["representation_seconds"] = perf_counter() - start

        start = perf_counter()
        model = build_model(config, k).fit(matrix, features)
        stage_times["fit_seconds"] = perf_counter() - start

        start = perf_counter()
        evaluation_matrix = vectorizer.transform(evaluation_data[text].astype(str))
        raw_weights = model.estimator_.transform(evaluation_matrix)
        stage_times[f"{stage}_transform_seconds"] = perf_counter() - start

        start = perf_counter()
        metrics, per_topic, distributions = evaluate_model(
            model, evaluation_matrix, features,
            int(config["topic_selection"]["coherence_top_n"]), raw_weights=raw_weights,
        )
        metrics["adjusted_rand_index"] = float(adjusted_rand_score(
            evaluation_data[gold], distributions.argmax(axis=1),
        ))
        metrics["normalized_mutual_info"] = float(normalized_mutual_info_score(
            evaluation_data[gold], distributions.argmax(axis=1),
        ))
        stage_times["evaluation_seconds"] = perf_counter() - start
        fit_split = "train" if stage == "validation" else "train+val"
        metrics.update({"fit_split": fit_split, "evaluation_split": stage})
        write_json(result_dir / f"{stage}_metrics.json", metrics)
        metric_rows.append({
            "split": stage, "fit_split": fit_split, "run_id": run_id,
            "ari": metrics["adjusted_rand_index"], "nmi": metrics["normalized_mutual_info"],
            "npmi": metrics["npmi_coherence"],
            "reconstruction_error": metrics["reconstruction_error"],
            "normalized_reconstruction_error": metrics["normalized_reconstruction_error"],
            "topic_diversity": metrics["topic_diversity"],
        })
        for name, duration in stage_times.items():
            phases[name] += duration
        timing_rows.append({"stage": stage, "fit_split": fit_split, **stage_times})

        docs, keywords, _, cases = build_document_outputs(
            evaluation_data, distributions, model, config, per_topic,
        )
        if stage == "validation":
            docs.to_parquet(artifact_dir / "validation_document_topics.parquet", index=False)
        else:
            dev_docs, _, representatives, _ = build_document_outputs(
                development, model.training_topic_distributions(), model, config, per_topic,
            )
            dev_docs.to_parquet(artifact_dir / "development_document_topics.parquet", index=False)
            docs.to_parquet(artifact_dir / "document_topics.parquet", index=False)
            (artifact_dir / "model").mkdir()
            joblib.dump({"vectorizer": vectorizer, "topic_model": model},
                        artifact_dir / "model/nmf_pipeline.joblib")
            pd.DataFrame(model.topic_word_weights_, columns=features).to_parquet(
                artifact_dir / "topic_word_matrix.parquet", index=False,
            )
            write_json(artifact_dir / "topic_keywords.json", keywords)
            write_json(artifact_dir / "representative_docs.json", representatives)
            cases.to_csv(result_dir / "multi_topic_cases.csv", index=False, encoding="utf-8-sig")

    runtime_scope = (
        "fixed_k_validation_train_fit_and_final_train_val_fit; "
        "representation=two_fit_transform; transform=vectorizer_plus_nmf; "
        "evaluation=metrics_only; excludes_dataset_io_artifact_export_and_ui"
    )
    runtime = {**phases, "runtime_seconds": sum(phases.values()), "runtime_scope": runtime_scope}
    runtime["environment"] = {
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "thread_limits": {name: os.environ.get(name) for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        )},
    }
    pd.DataFrame(timing_rows).to_csv(result_dir / "runtime_by_stage.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(result_dir / "split_metrics.csv", index=False)
    flat_metrics = {
        f"{row['split']}_{name}": row[name]
        for row in metric_rows for name in (
            "ari", "nmi", "npmi", "reconstruction_error",
            "normalized_reconstruction_error", "topic_diversity",
        )
    }
    write_json(result_dir / "dashboard_metrics.json", {**flat_metrics, **runtime})
    write_json(result_dir / "runtime.json", {**runtime, "stages": timing_rows})
    metadata = base_metadata(run_id, "dashboard", dataset, config)
    metadata.update({
        "selected_n_topics": k, "selection_run_id": config["topic_selection"].get("selection_run_id"),
        "fit_split": "train+val", "evaluation_split": "test", "test_accessed": True,
        "fixed_k_remeasurement": True, "representative_split": "train+val",
        "model_file": "model/nmf_pipeline.joblib", **runtime,
    })
    write_json(artifact_dir / "config.json", config)
    write_json(artifact_dir / "run_metadata.json", metadata)
    return artifact_dir, result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nmf.yaml")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_yaml(resolve_project_path(args.config))
    dataset, _ = load_dataset_v1(config)
    for path in prepare_dashboard(config, dataset, args.run_id):
        print(path)


if __name__ == "__main__":
    main()
