"""Check that stability ignores topic numbering and keeps held-out data out of fit."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.nmf.run_experiment import load_yaml, sha256_file
from experiments.nmf.run_stability import evaluate_pairs, run_stability, validate_settings


def test_topic_permutation_does_not_change_stability():
    distributions = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    topics = [["hospital", "doctor"], ["school", "student"]]
    results = {
        42: {"topics": topics, "distributions": distributions},
        52: {"topics": topics[::-1], "distributions": distributions[:, ::-1]},
        62: {"topics": topics, "distributions": distributions},
    }
    assignments = pd.concat([
        pd.DataFrame({"seed": seed, "article_id": ["b", "a", "c"],
                      "topic_id": item["distributions"].argmax(axis=1)}).sample(frac=1, random_state=seed)
        for seed, item in results.items()
    ], ignore_index=True)
    pairs, matches = evaluate_pairs(results, assignments)
    assert len(pairs) == 3
    assert len(matches) == 6
    assert (pairs.seed_a < pairs.seed_b).all()
    np.testing.assert_allclose(pairs[["ari_between_runs", "keyword_jaccard", "document_cosine"]], 1.0)
    swapped = matches.loc[matches.seed_a.eq(42) & matches.seed_b.eq(52)]
    assert list(zip(swapped.topic_a, swapped.topic_b)) == [(0, 1), (1, 0)]


def test_stability_run_fits_train_only_and_exports_aligned_e04_data(tmp_path):
    texts = [
        "hospital doctor nurse medicine", "doctor medicine hospital care",
        "nurse care hospital doctor", "medicine hospital health nurse",
        "school student teacher classroom", "teacher school education student",
        "student classroom school education", "education teacher classroom school",
        "doctor hospital validation_only", "school student validation_only",
        "nurse doctor medicine", "validation_only",
        "heldout_only secret_test", "heldout_only secret_test",
    ]
    ids = [f"article_{i}" for i in range(len(texts))]
    dataset = pd.DataFrame({
        "article_id": ids, "text": texts, "title": "Example",
        "published_at": "2026-01-01", "source": "fixture",
    })
    data_path = tmp_path / "dataset.csv"
    dataset.to_csv(data_path, index=False)
    split_path = tmp_path / "splits.csv"
    pd.DataFrame({"article_id": ids, "split": ["train"] * 8 + ["val"] * 4 + ["test"] * 2}).to_csv(split_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "review_status": "approved", "dataset_version": "v1.0",
        "variants": {"p3_no_stopwords": {"sha256": sha256_file(data_path)}},
    }), encoding="utf-8")
    base = load_yaml(Path("configs/nmf.yaml"))
    base["data"].update(dataset_path=str(data_path), split_manifest_path=str(split_path),
                        dataset_manifest_path=str(manifest_path))
    base["representation"].update(ngram_range=[1, 1], min_df=1, max_df=1.0)
    base["topic_selection"]["selected_n_topics"] = 2
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    config = {
        "base_config": str(base_path),
        "stability": {"n_topics": 2, "random_seeds": [42, 52, 62], "top_n_words": 2,
                      "fit_split": "train", "evaluation_split": "val", "thread_limit": 1},
        "outputs": {"directory": str(tmp_path / "result")},
    }
    config_path = tmp_path / "stability.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = run_stability(config_path)
    vocabulary = json.loads((output / "feature_names.json").read_text(encoding="utf-8"))
    assert "validation_only" not in vocabulary
    assert "heldout_only" not in vocabulary
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["n_train"] == 8
    assert metadata["n_validation"] == 4
    assert metadata["test_used_for_fit_or_evaluation"] is False
    assignments = pd.read_csv(output / "validation_assignments.csv")
    assert len(assignments) == 12
    assert not assignments.duplicated(["seed", "article_id"]).any()
    assert set(assignments.article_id) == set(ids[8:12])
    assert assignments.loc[assignments.article_id.eq("article_11"), "topic_id"].eq(-1).all()
    per_seed = pd.read_csv(output / "per_seed.csv")
    np.testing.assert_allclose(per_seed.coverage, 0.75)
    np.testing.assert_allclose(per_seed.outlier_rate, 0.25)
    with np.load(output / "validation_distributions.npz") as saved:
        saved_ids = list(saved["article_ids"])
        assert set(saved_ids) == set(ids[8:12])
        zero_index = saved_ids.index("article_11")
        for seed in [42, 52, 62]:
            weights = saved[f"seed_{seed}"]
            assert weights.shape == (4, 2)
            nonzero = np.arange(4) != zero_index
            np.testing.assert_allclose(weights[nonzero].sum(axis=1), 1.0)
            np.testing.assert_allclose(weights[zero_index], 0.0)
            expected_labels = weights.argmax(axis=1)
            expected_labels[~nonzero] = -1
            actual = assignments.loc[assignments.seed.eq(seed)].set_index("article_id").loc[saved_ids, "topic_id"]
            np.testing.assert_array_equal(actual, expected_labels)
    for filename, expected in metadata["output_sha256"].items():
        assert sha256_file(output / filename) == expected
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        run_stability(config_path)


@pytest.mark.parametrize("change", [
    {"random_seeds": [42, 42, 52]}, {"random_seeds": [42, 52]},
    {"random_seeds": [42, 52, -1]}, {"random_seeds": [42, 52, True]},
    {"evaluation_split": "test"}, {"fit_split": "train+val"}, {"n_topics": 12},
])
def test_invalid_protocol_is_rejected(change):
    config = {"stability": {
        "n_topics": 10, "random_seeds": [42, 52, 62], "top_n_words": 10,
        "fit_split": "train", "evaluation_split": "val", "thread_limit": 1,
    }}
    settings = deepcopy(config)
    settings["stability"].update(change)
    with pytest.raises(ValueError):
        validate_settings(settings, {"topic_selection": {"selected_n_topics": 10}})
