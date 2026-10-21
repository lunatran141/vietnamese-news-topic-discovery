
"""Tests for leader-requested LDA Common Artifact extensions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.artifacts.loader import load_artifact_bundle


# Validate the successful extended export before updating the main run.
ROOT = Path(
    "artifacts/common/lda/lda_k15_seed42"
)


@pytest.fixture(scope="module")
def artifact():
    assert ROOT.is_dir(), f"Missing artifact: {ROOT}"

    bundle = load_artifact_bundle(ROOT)

    summary = json.loads(
        (ROOT / "model_summary.json").read_text(
            encoding="utf-8"
        )
    )

    return bundle, summary


def test_all_eight_files():
    required = {
        "manifest.json",
        "model_summary.json",
        "config_used.yaml",
        "document_topics.csv",
        "topic_summary.csv",
        "topic_keywords.csv",
        "representative_docs.csv",
        "projection_2d.csv",
    }

    assert all((ROOT / name).is_file() for name in required)


def test_extended_document_columns(artifact):
    bundle, _ = artifact
    docs = bundle.document_topics

    required = {
        "article_id",
        "split",
        "topic_id",
        "topic_probability",
        "second_topic_id",
        "second_topic_probability",
        "top_two_share_gap",
        "topic_entropy",
        "is_multi_topic",
        "title",
        "category_gold",
    }

    assert required.issubset(docs.columns)
    assert len(docs) == 9274


def test_multi_topic_definition(artifact):
    bundle, summary = artifact
    docs = bundle.document_topics

    threshold = summary["multi_topic"]["threshold"]

    second_probs = pd.to_numeric(
        docs["second_topic_probability"]
    )

    actual = docs["is_multi_topic"].astype(str).str.lower()
    expected = (
        second_probs >= threshold
    ).astype(str).str.lower()

    assert threshold == pytest.approx(0.20)
    assert (actual == expected).all()
    assert int((second_probs >= threshold).sum()) == 4574


def test_top_two_probabilities(artifact):
    bundle, _ = artifact
    docs = bundle.document_topics

    first = pd.to_numeric(docs["topic_probability"])
    second = pd.to_numeric(
        docs["second_topic_probability"]
    )
    gap = pd.to_numeric(docs["top_two_share_gap"])
    entropy = pd.to_numeric(docs["topic_entropy"])

    assert first.between(0, 1).all()
    assert second.between(0, 1).all()
    assert (first >= second).all()

    np.testing.assert_allclose(
        gap.to_numpy(),
        (first - second).to_numpy(),
        rtol=0,
        atol=1e-8,
    )

    assert np.isfinite(entropy).all()
    assert (entropy >= 0).all()
    assert (entropy <= np.log(15) + 1e-8).all()


def test_runtime_fields(artifact):
    _, summary = artifact

    required = [
        "runtime_seconds",
        "representation_seconds",
        "fit_seconds",
        "validation_inference_seconds",
        "test_inference_seconds",
        "evaluation_seconds",
        "runtime_mean",
        "runtime_std",
        "runtime_scope",
    ]

    assert all(key in summary for key in required)

    for key in required:
        if key == "runtime_scope":
            assert isinstance(summary[key], str)
            assert summary[key].strip()
        else:
            assert summary[key] is not None
            assert np.isfinite(float(summary[key]))
            assert float(summary[key]) >= 0

    assert summary["runtime_seconds"] == pytest.approx(
        22.3437
    )
    assert summary["runtime_std"] == pytest.approx(
        0.3521
    )


def test_validation_and_test_metrics(artifact):
    _, summary = artifact
    metrics = summary["metrics"]

    expected = {
        "validation_ari": 0.41522628538438183,
        "validation_nmi": 0.5594385556458895,
        "validation_npmi": 0.1343512695067231,
        "test_ari": 0.40646609063623057,
        "test_nmi": 0.5727127152837649,
        "test_npmi": 0.14709446410016874,
        "perplexity": 1783.2125727666962,
        "topic_diversity": 0.6533333333333333,
        "stability_mean": 0.4433506838305599,
        "stability_std": 0.03405692541534584,
    }

    for key, value in expected.items():
        assert key in metrics
        assert metrics[key] == pytest.approx(value)


def test_evaluation_protocol(artifact):
    _, summary = artifact

    protocol = summary["evaluation_protocol"]

    assert protocol["validation"]["fit_split"] == "train"
    assert protocol["validation"]["evaluation_split"] == "val"

    assert protocol["test"]["fit_split"] == "train+val"
    assert protocol["test"]["evaluation_split"] == "test"

    assert (
        protocol["validation"]["npmi_definition"]
        == "document_binary_common"
    )

    assert (
        protocol["stability"]["metric"]
        == "matched_jaccard"
    )
    assert protocol["stability"]["reference_seed"] == 42
    assert protocol["stability"]["candidate_seeds"] == [
        52, 62, 72, 82
    ]


def test_per_seed_metrics_csv():
    path = Path("results/lda/m09/stability_per_seed.csv")
    df = pd.read_csv(path)

    required = {
        "seed",
        "topic_count",
        "ari",
        "nmi",
        "npmi",
        "topic_diversity",
        "runtime_seconds",
    }

    assert required.issubset(df.columns)
    assert len(df) == 5
    assert df["seed"].tolist() == [42, 52, 62, 72, 82]
    assert (df["topic_count"] == 15).all()
    assert np.isfinite(
        df[
            [
                "ari",
                "nmi",
                "npmi",
                "topic_diversity",
                "runtime_seconds",
            ]
        ].to_numpy(dtype=float)
    ).all()
