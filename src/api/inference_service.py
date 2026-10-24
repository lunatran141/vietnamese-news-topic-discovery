"""Inference BERTopic từ model và artifact đã lưu, không huấn luyện lại."""

from __future__ import annotations

import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.api.schemas import PredictResponse, RepresentativeArticle
from src.preprocessing.clean_text import clean_text_p0
from src.preprocessing.normalize_text import normalize_text_p1
from src.preprocessing.remove_stopwords import load_stopwords, remove_stopwords_p3


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "e1_vietnamese_sbert"
COMMON_ARTIFACT_DIR = (
    REPO_ROOT / "artifacts" / "common" / "bertopic" / RUN_ID
)
DEFAULT_MODEL_CANDIDATES = (
    REPO_ROOT / "artifacts" / "bertopic" / "final" / RUN_ID / "model",
    REPO_ROOT
    / "artifacts"
    / "bertopic"
    / "experiments"
    / RUN_ID
    / "model",
)


class InferenceUnavailableError(RuntimeError):
    """Model hoặc artifact cần cho inference chưa sẵn sàng."""


def preprocess_article(title: str, content: str) -> str:
    """Áp dụng đúng chuỗi P0 -> P1 và ghép title + content như E1."""
    clean_title = clean_text_p0(title)
    clean_content = clean_text_p0(content)
    normalized_title = normalize_text_p1(clean_title)
    normalized_content = normalize_text_p1(
        clean_content,
        title=clean_title,
        config_path=REPO_ROOT / "configs" / "preprocessing.yaml",
    )
    text = " ".join(
        part for part in (normalized_title, normalized_content) if part
    ).strip()
    if not text:
        raise ValueError("Tiêu đề và nội dung không được đồng thời để trống.")
    return text


def _resolve_model_path() -> Path:
    configured = os.getenv("BERTOPIC_MODEL_PATH")
    candidates = (Path(configured),) if configured else DEFAULT_MODEL_CANDIDATES
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else REPO_ROOT / candidate
        if path.exists():
            return path
    expected = ", ".join(str(path) for path in candidates)
    raise InferenceUnavailableError(
        "Không tìm thấy model BERTopic đã lưu. "
        f"Các đường dẫn đã kiểm tra: {expected}"
    )


def _assigned_confidence(probabilities: Any) -> float | None:
    if probabilities is None:
        return None
    values = np.asarray(probabilities, dtype=float)
    if values.size == 0:
        return None
    row = values[0] if values.ndim > 1 else values
    finite = np.asarray(row, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    value = float(np.max(finite))
    return min(max(value, 0.0), 1.0)


class BERTopicInferenceService:
    """Lazy-load model một lần và phục vụ nhiều request dự đoán."""

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        artifact_dir: Path = COMMON_ARTIFACT_DIR,
        model: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.artifact_dir = artifact_dir
        self._model = model
        self._labels: dict[int, dict[str, Any]] | None = None
        self._keywords: pd.DataFrame | None = None
        self._representatives: pd.DataFrame | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        path = self.model_path or _resolve_model_path()
        try:
            from bertopic import BERTopic

            self._model = BERTopic.load(str(path))
        except Exception as exc:  # pragma: no cover - phụ thuộc model thật
            raise InferenceUnavailableError(
                f"Không thể tải model BERTopic tại {path}: {exc}"
            ) from exc
        self.model_path = path
        return self._model

    def _load_artifacts(self) -> None:
        if self._labels is not None:
            return
        mapping_path = self.artifact_dir / "topic_display_mapping.json"
        keywords_path = self.artifact_dir / "topic_keywords.csv"
        representatives_path = self.artifact_dir / "representative_docs.csv"
        missing = [
            str(path)
            for path in (mapping_path, keywords_path, representatives_path)
            if not path.is_file()
        ]
        if missing:
            raise InferenceUnavailableError(
                "Thiếu artifact giải thích kết quả: " + ", ".join(missing)
            )
        with mapping_path.open(encoding="utf-8") as file:
            raw_mapping = json.load(file)
        self._labels = {int(key): value for key, value in raw_mapping.items()}
        self._keywords = pd.read_csv(keywords_path, encoding="utf-8-sig")
        self._representatives = pd.read_csv(
            representatives_path, encoding="utf-8-sig"
        )

    def _topic_name(self, topic_id: int) -> str:
        assert self._labels is not None
        item = self._labels.get(topic_id, {})
        return str(
            item.get("display_label")
            or item.get("final_label")
            or item.get("suggested_label")
            or ("Outlier / Chưa xác định" if topic_id == -1 else f"Topic {topic_id}")
        )

    def _topic_keywords(self, topic_id: int, limit: int = 10) -> list[str]:
        assert self._keywords is not None
        rows = self._keywords.loc[
            pd.to_numeric(self._keywords["topic_id"], errors="coerce").eq(topic_id)
        ].copy()
        if "rank" in rows:
            rows = rows.sort_values("rank", kind="stable")
        return rows["keyword"].dropna().astype(str).head(limit).tolist()

    def _representative_articles(
        self, topic_id: int, limit: int = 3
    ) -> list[RepresentativeArticle]:
        assert self._representatives is not None
        rows = self._representatives.loc[
            pd.to_numeric(
                self._representatives["topic_id"], errors="coerce"
            ).eq(topic_id)
        ].copy()
        if "rank" in rows:
            rows = rows.sort_values("rank", kind="stable")
        articles: list[RepresentativeArticle] = []
        for record in rows.head(limit).to_dict("records"):
            split = record.get("split")
            articles.append(
                RepresentativeArticle(
                    article_id=str(record.get("article_id", "")),
                    title=str(record.get("title", "")),
                    split=None if pd.isna(split) else str(split),
                )
            )
        return articles

    def predict(self, title: str, content: str) -> PredictResponse:
        text = preprocess_article(title, content)
        model = self._load_model()
        self._load_artifacts()
        started = time.perf_counter()
        try:
            topics, probabilities = model.transform([text])
        except Exception as exc:
            raise InferenceUnavailableError(
                f"Model BERTopic không thể dự đoán: {exc}"
            ) from exc
        elapsed = time.perf_counter() - started
        topic_id = int(np.asarray(topics).reshape(-1)[0])
        confidence = _assigned_confidence(probabilities)
        if confidence is not None and not math.isfinite(confidence):
            confidence = None
        return PredictResponse(
            model="BERTopic",
            run_id=RUN_ID,
            topic_id=topic_id,
            topic_name=self._topic_name(topic_id),
            is_outlier=topic_id == -1,
            confidence=confidence,
            keywords=self._topic_keywords(topic_id),
            representative_articles=self._representative_articles(topic_id),
            inference_seconds=round(elapsed, 4),
            preprocessing_variant="p1_normalized",
            inference_mode="saved_model_transform",
        )


@lru_cache(maxsize=1)
def get_inference_service() -> BERTopicInferenceService:
    """Dùng chung model đã lưu; không fit trên request."""
    return MultiModelInferenceService()


MODEL_RUNS = {
    "bertopic": RUN_ID,
    "hdbscan": "minilm_title_content_eom_15_5",
    "kmeans": "kmeans_baseline",
    "nmf": "nmf_p3_k10_seed42_dashboard_v2",
    "lda": "lda_k15_seed42",
}


class MultiModelInferenceService:
    """Inference từ năm pipeline cố định và common artifact tương ứng."""

    def __init__(self, bertopic: BERTopicInferenceService | None = None) -> None:
        self.bertopic = bertopic or BERTopicInferenceService()
        self._models: dict[str, Any] = {}
        self._artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    @staticmethod
    def _method(model: str) -> str:
        method = model.strip().lower()
        if method not in MODEL_RUNS:
            raise ValueError(f"Mô hình không hợp lệ: {model}. Chọn: {', '.join(MODEL_RUNS)}")
        return method

    def _common(self, method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if method not in self._artifacts:
            root = REPO_ROOT / "artifacts" / "common" / method / MODEL_RUNS[method]
            paths = [root / name for name in ("topic_summary.csv", "topic_keywords.csv", "representative_docs.csv")]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise InferenceUnavailableError("Thiếu common artifact: " + ", ".join(missing))
            self._artifacts[method] = tuple(pd.read_csv(path, encoding="utf-8-sig") for path in paths)
        return self._artifacts[method]

    def topics(self, model: str = "bertopic") -> dict[str, Any]:
        method = self._method(model)
        summary, keywords, _ = self._common(method)
        labels = self._labels(method)
        rows = []
        for record in summary.to_dict("records"):
            topic_id = int(record["topic_id"])
            rows.append({
                "topic_id": topic_id,
                "topic_name": labels.get(topic_id) or str(record.get("topic_name") or f"Topic {topic_id}"),
                "document_count": int(record.get("document_count") or 0),
                "keywords": self._keywords(keywords, topic_id),
            })
        return {"model": method, "run_id": MODEL_RUNS[method], "topics": rows}

    @staticmethod
    def _keywords(frame: pd.DataFrame, topic_id: int) -> list[str]:
        rows = frame.loc[pd.to_numeric(frame["topic_id"], errors="coerce").eq(topic_id)]
        if "rank" in rows:
            rows = rows.sort_values("rank", kind="stable")
        return rows["keyword"].dropna().astype(str).head(10).tolist()

    def _labels(self, method: str) -> dict[int, str]:
        path = REPO_ROOT / "artifacts" / "common" / method / MODEL_RUNS[method] / "topic_display_mapping.json"
        if not path.is_file():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = raw.get("mapping", raw)
        return {
            int(key): str(value.get("display_label") or value.get("display_name") or value.get("final_label") or value.get("suggested_label") or "")
            for key, value in entries.items() if isinstance(value, dict)
        }

    def _load(self, method: str) -> Any:
        if method in self._models:
            return self._models[method]
        import joblib
        paths = {
            "kmeans": REPO_ROOT / "artifacts/kmeans/kmeans_model.joblib",
            "nmf": REPO_ROOT / "artifacts/nmf/nmf_p3_k10_seed42_dashboard_v2/model/nmf_pipeline.joblib",
            "lda": REPO_ROOT / "artifacts/lda/final/lda_pipeline.joblib",
        }
        if method in paths:
            if not paths[method].is_file():
                raise InferenceUnavailableError(f"Thiếu model đã lưu: {paths[method]}")
            self._models[method] = joblib.load(paths[method])
        elif method == "hdbscan":
            root = REPO_ROOT / "artifacts/hdbscan/validation"
            reducer_path = root / "minilm_title_content_umap.joblib"
            clusterer_path = root / "minilm_title_content_eom_15_5_hdbscan.joblib"
            if not reducer_path.is_file() or not clusterer_path.is_file():
                raise InferenceUnavailableError("Thiếu UMAP/HDBSCAN đã lưu trong artifacts/hdbscan/validation")
            try:
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            except Exception as exc:
                raise InferenceUnavailableError(f"Không tải được encoder MiniLM: {exc}") from exc
            self._models[method] = (encoder, joblib.load(reducer_path), joblib.load(clusterer_path))
        return self._models[method]

    @staticmethod
    def _lexical_text(title: str, content: str, method: str) -> str:
        p0 = clean_text_p0(f"{title}. {content}".strip(" ."))
        p1 = normalize_text_p1(p0)
        try:
            from pyvi import ViTokenizer
        except ImportError as exc:
            raise InferenceUnavailableError("Thiếu pyvi để tái tạo P2/P3") from exc
        p2 = ViTokenizer.tokenize(p1)
        if method != "nmf":
            return p2
        stopwords = load_stopwords(REPO_ROOT / "artifacts/configs/vietnamese_stopwords.txt")
        return remove_stopwords_p3(p2, stopwords)

    def predict(self, title: str, content: str, model: str = "bertopic") -> PredictResponse:
        method = self._method(model)
        if not title.strip() and not content.strip():
            raise ValueError("Tiêu đề và nội dung không được đồng thời để trống.")
        if method == "bertopic":
            return self.bertopic.predict(title, content)
        summary, keywords, representatives = self._common(method)
        pipeline = self._load(method)
        started = time.perf_counter()
        try:
            if method == "hdbscan":
                from hdbscan.prediction import approximate_predict
                encoder, reducer, clusterer = pipeline
                short_content = " ".join(clean_text_p0(content).split()[:60])
                source = f"{clean_text_p0(title)}. {short_content}".strip(" .")
                vector = encoder.encode([source], normalize_embeddings=True, show_progress_bar=False)
                label, strength = approximate_predict(clusterer, reducer.transform(vector))
                topic_id, confidence = int(label[0]), float(strength[0])
                variant = "p0_minimal"
            else:
                source = self._lexical_text(title, content, method)
                if method == "kmeans":
                    vectorizer_path = REPO_ROOT / "artifacts/tfidf/tfidf_vectorizer.joblib"
                    import joblib
                    vectorizer = self._models.setdefault("kmeans_vectorizer", joblib.load(vectorizer_path))
                    matrix = vectorizer.transform([source])
                    topic_id = int(pipeline.predict(matrix)[0])
                    confidence = None  # KMeans không có xác suất đã hiệu chỉnh.
                    variant = "p2_segmented"
                else:
                    key = "topic_model" if method == "nmf" else "lda_model"
                    matrix = pipeline["vectorizer"].transform([source])
                    weights = np.asarray(pipeline[key].transform(matrix))[0]
                    topic_id = int(np.argmax(weights))
                    confidence = float(weights[topic_id] / weights.sum()) if weights.sum() > 0 else None
                    variant = "p3_no_stopwords" if method == "nmf" else "p2_segmented"
        except InferenceUnavailableError:
            raise
        except Exception as exc:
            raise InferenceUnavailableError(f"Không thể suy diễn {method}: {exc}") from exc
        elapsed = time.perf_counter() - started
        selected = summary.loc[pd.to_numeric(summary["topic_id"], errors="coerce").eq(topic_id)]
        base_name = str(selected.iloc[0].get("topic_name") or f"Topic {topic_id}") if not selected.empty else f"Topic {topic_id}"
        articles = representatives.loc[pd.to_numeric(representatives["topic_id"], errors="coerce").eq(topic_id)]
        if "rank" in articles:
            articles = articles.sort_values("rank", kind="stable")
        examples = [RepresentativeArticle(article_id=str(row.get("article_id", "")), title=str(row.get("title", "")), split=None if pd.isna(row.get("split")) else str(row.get("split"))) for row in articles.head(3).to_dict("records")]
        return PredictResponse(
            model={"hdbscan": "HDBSCAN", "kmeans": "KMeans", "nmf": "NMF", "lda": "LDA"}[method],
            run_id=MODEL_RUNS[method], topic_id=topic_id,
            topic_name=self._labels(method).get(topic_id) or ("Outlier / Chưa xác định" if topic_id == -1 else base_name),
            is_outlier=topic_id == -1, confidence=confidence,
            keywords=self._keywords(keywords, topic_id), representative_articles=examples,
            inference_seconds=round(elapsed, 4), preprocessing_variant=variant,
            inference_mode="saved_model_transform",
        )

