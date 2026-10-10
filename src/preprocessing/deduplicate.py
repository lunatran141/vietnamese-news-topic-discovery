"""
Module khử trùng lặp dữ liệu (Task D02 — Deduplication Pipeline).

Xây dựng pipeline khử trùng lặp 2 tầng:
  - Tầng 1: Exact duplicate (URL, content hash, normalized title)
  - Tầng 2: Near duplicate (MinHash LSH + Jaccard similarity)

Cách chạy:
    python -m src.preprocessing.deduplicate
    python -m src.preprocessing.deduplicate --input data/interim/news_articles_validated.csv

"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV      = ROOT / "data" / "interim" / "news_articles_validated.csv"
OUTPUT_DEDUP   = ROOT / "data" / "interim" / "news_articles_deduplicated.csv"
OUTPUT_REMOVED = ROOT / "results" / "dedup_data" / "dedup_removed.csv"
OUTPUT_CAND    = ROOT / "results" / "dedup_data" / "dedup_candidates_review.csv"
OUTPUT_BA      = ROOT / "results" / "dedup_data" / "dedup_before_after.csv"

CONTENT_SIM_REMOVE: float = 0.85
CONTENT_SIM_MANUAL_LOW: float = 0.80
TITLE_SIM_HIGH: float = 0.60
TITLE_CONTENT_THRESH: float = 0.70
NUM_PERM: int = 128
LSH_THRESHOLD: float = 0.70

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deduplicate")


def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản: NFC, lower, bỏ dấu câu, gộp khoảng trắng."""
    if not isinstance(text, str) or not text.strip():
        return ""
    norm = unicodedata.normalize("NFC", text).lower()
    norm = re.sub(r"[^\w\s]", " ", norm, flags=re.UNICODE)
    return re.sub(r"\s+", " ", norm).strip()


def hash_content(text: str) -> str:
    """SHA-256 hash nội dung đã chuẩn hóa."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def normalize_url(url: str) -> str:
    """Chuẩn hóa URL: bỏ scheme, www., query params, trailing slash."""
    if not isinstance(url, str) or not url.strip():
        return ""
    try:
        p = urlparse(url)
    except Exception:
        return url.strip().lower()
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return f"{netloc}{p.path.rstrip('/')}"


def jaccard_sets(set_a: set, set_b: set) -> float:
    """Jaccard similarity giữa hai tập hợp."""
    if not set_a and not set_b:
        return 0.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else 0.0


def tokenize_words(text: str) -> set:
    """Tách văn bản đã chuẩn hóa thành tập word tokens."""
    norm = normalize_text(text)
    return set(norm.split()) if norm else set()


def build_minhash(text: str, num_perm: int = NUM_PERM) -> MinHash:
    """Tạo MinHash signature từ word tokens của văn bản chuẩn hóa."""
    m = MinHash(num_perm=num_perm, seed=42)
    norm = normalize_text(text)
    words = norm.split()
    if not words:
        m.update(b"")
    else:
        for s in set(words):
            m.update(s.encode("utf-8"))
    return m


def pick_keeper(group_df: pd.DataFrame) -> str:
    """
    Chọn bài giữ lại theo thứ tự ưu tiên:
      1. Nội dung dài hơn (len(content) giảm dần)
      2. Ngày đăng hợp lệ (published_at not NaT)
      3. article_id nhỏ hơn (ổn định)
    Tuyệt đối không dùng category_gold.
    """
    scored = group_df[["article_id", "content", "published_at"]].copy()
    scored["_clen"] = scored["content"].fillna("").str.len()
    scored["_has_date"] = (
        pd.to_datetime(scored["published_at"], errors="coerce", utc=True)
        .notna()
        .astype(int)
    )
    scored = scored.sort_values(
        by=["_clen", "_has_date", "article_id"],
        ascending=[False, False, True],
    )
    return str(scored.iloc[0]["article_id"])


class UnionFind:
    """Disjoint-set Union-Find với path compression và union by rank."""

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def clusters(self) -> Dict[str, Set[str]]:
        groups: Dict[str, Set[str]] = defaultdict(set)
        for x in self._parent:
            groups[self.find(x)].add(x)
        return dict(groups)


def step1_exact_dedup(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Exact dedup tuần tự: URL -> content hash -> normalized title."""
    logger.info("Tầng 1: Exact Duplicate")

    df = df.copy()
    df["_norm_url"] = df["url"].apply(normalize_url)
    df["_hash_content"] = df["content"].apply(hash_content)
    df["_norm_title"] = df["title"].apply(normalize_text)

    removed: List[Dict[str, Any]] = []
    remaining: Set[str] = set(df["article_id"].astype(str))

    dedup_keys = [
        ("_norm_url", "exact_url"),
        ("_hash_content", "exact_content"),
        ("_norm_title", "exact_title"),
    ]

    for key_col, dup_type in dedup_keys:
        working = df[df["article_id"].astype(str).isin(remaining)]
        step_removed = 0

        for key_val, grp in working.groupby(key_col, sort=False):
            if len(grp) <= 1:
                continue
            keeper_id = pick_keeper(grp)
            for _, row in grp.iterrows():
                aid = str(row["article_id"])
                if aid == keeper_id:
                    continue
                ev_val = str(key_val)
                if dup_type == "exact_content":
                    ev_val = f"sha256:{ev_val[:16]}..."
                removed.append({
                    "kept_article_id": keeper_id,
                    "removed_article_id": aid,
                    "duplicate_type": dup_type,
                    "evidence": json.dumps({dup_type: ev_val}, ensure_ascii=False),
                })
                remaining.discard(aid)
                step_removed += 1

        logger.info("%s: loại %d bài (còn %d)", dup_type, step_removed, len(remaining))

    df_clean = (
        df[df["article_id"].astype(str).isin(remaining)]
        .drop(columns=["_norm_url", "_hash_content", "_norm_title"])
        .copy()
    )
    logger.info("Tầng 1 hoàn tất: %d -> %d bài", len(df), len(df_clean))
    return df_clean, removed


def step2_near_dedup(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], pd.DataFrame]:
    """Near dedup với MinHash LSH và Dual-scoring."""
    logger.info("Tầng 2: Near Duplicate")

    ids = df["article_id"].astype(str).tolist()
    contents = dict(zip(df["article_id"].astype(str), df["content"].fillna("")))
    titles = dict(zip(df["article_id"].astype(str), df["title"].fillna("")))

    logger.info("Sinh MinHash signatures...")
    cmh: Dict[str, MinHash] = {}
    for aid in ids:
        cmh[aid] = build_minhash(contents[aid])

    logger.info("Xây dựng LSH index...")
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    for aid in ids:
        try:
            lsh.insert(aid, cmh[aid])
        except ValueError:
            pass

    candidate_pairs: Set[Tuple[str, str]] = set()
    for aid in ids:
        for other in lsh.query(cmh[aid]):
            if aid < other:
                candidate_pairs.add((aid, other))

    logger.info("Tìm được %d cặp ứng viên", len(candidate_pairs))

    ttok: Dict[str, set] = {aid: tokenize_words(titles[aid]) for aid in ids}

    cand_records: List[Dict[str, Any]] = []
    remove_pairs: List[Tuple[str, str]] = []

    for aid1, aid2 in sorted(candidate_pairs):
        c_sim = round(cmh[aid1].jaccard(cmh[aid2]), 4)
        t_sim = round(jaccard_sets(ttok[aid1], ttok[aid2]), 4)

        if c_sim >= CONTENT_SIM_REMOVE:
            decision = "remove"
            reason = f"content_sim={c_sim} >= {CONTENT_SIM_REMOVE}"
        elif t_sim >= TITLE_SIM_HIGH and c_sim >= TITLE_CONTENT_THRESH:
            decision = "remove"
            reason = f"title_sim={t_sim} >= {TITLE_SIM_HIGH} and content_sim={c_sim} >= {TITLE_CONTENT_THRESH}"
        elif c_sim >= CONTENT_SIM_MANUAL_LOW:
            decision = "manual_review"
            reason = f"content_sim={c_sim} in [{CONTENT_SIM_MANUAL_LOW}, {CONTENT_SIM_REMOVE})"
        else:
            decision = "keep_both"
            reason = f"content_sim={c_sim} < {CONTENT_SIM_MANUAL_LOW}"

        cand_records.append({
            "article_id_1": aid1,
            "article_id_2": aid2,
            "title_similarity": t_sim,
            "content_similarity": c_sim,
            "decision": decision,
            "reason": reason,
        })
        if decision == "remove":
            remove_pairs.append((aid1, aid2))

    n_remove = sum(1 for r in cand_records if r["decision"] == "remove")
    n_manual = sum(1 for r in cand_records if r["decision"] == "manual_review")
    n_keep = sum(1 for r in cand_records if r["decision"] == "keep_both")
    logger.info("Quyết định: remove=%d, manual_review=%d, keep_both=%d", n_remove, n_manual, n_keep)

    uf = UnionFind()
    for a, b in remove_pairs:
        uf.union(a, b)

    clusters = uf.clusters()
    near_removed: List[Dict[str, Any]] = []
    removed_aids: Set[str] = set()

    for _, members in sorted(clusters.items()):
        if len(members) <= 1:
            continue
        cluster_df = df[df["article_id"].astype(str).isin(members)]
        keeper_id = pick_keeper(cluster_df)
        for aid in sorted(members):
            if aid == keeper_id:
                continue
            sim_to_keeper = round(cmh[aid].jaccard(cmh[keeper_id]), 4)
            near_removed.append({
                "kept_article_id": keeper_id,
                "removed_article_id": aid,
                "duplicate_type": "near_duplicate",
                "evidence": json.dumps({"content_similarity_to_keeper": sim_to_keeper}, ensure_ascii=False),
            })
            removed_aids.add(aid)

    df_clean = df[~df["article_id"].astype(str).isin(removed_aids)].copy()

    if cand_records:
        candidates_df = pd.DataFrame(cand_records)
    else:
        candidates_df = pd.DataFrame(columns=[
            "article_id_1", "article_id_2", "title_similarity",
            "content_similarity", "decision", "reason",
        ])

    logger.info("Tầng 2 hoàn tất: loại %d bài (còn %d)", len(removed_aids), len(df_clean))
    return df_clean, near_removed, candidates_df


def compute_before_after(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    """Bảng phân bổ trước và sau theo source và category_gold."""
    before = df_before.groupby(["source", "category_gold"], sort=True).size().reset_index(name="count_before")
    after = df_after.groupby(["source", "category_gold"], sort=True).size().reset_index(name="count_after")
    result = before.merge(after, on=["source", "category_gold"], how="left")
    result["count_after"] = result["count_after"].fillna(0).astype(int)
    result["removed_count"] = result["count_before"] - result["count_after"]
    result["removal_rate_pct"] = ((result["removed_count"] / result["count_before"]) * 100).round(2)
    return result.sort_values(["source", "category_gold"]).reset_index(drop=True)


def run_pipeline(input_path: Path | None = None) -> Dict[str, Any]:
    """Thực thi toàn bộ pipeline khử trùng lặp và lưu các file kết quả."""
    input_path = input_path or INPUT_CSV
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    logger.info("Đọc dữ liệu: %s", input_path)
    df = pd.read_csv(input_path, dtype=str)
    df["article_id"] = df["article_id"].astype(str)
    total_before = len(df)

    df_after_exact, exact_removed = step1_exact_dedup(df)
    df_final, near_removed, candidates_df = step2_near_dedup(df_after_exact)

    all_removed = exact_removed + near_removed
    total_removed = len({r["removed_article_id"] for r in all_removed})
    total_after = len(df_final)

    stats: Dict[str, Any] = {
        "total_before": total_before,
        "total_after": total_after,
        "exact_removed": len(exact_removed),
        "near_removed": len(near_removed),
        "total_removed": total_removed,
        "removal_pct": (total_removed / total_before * 100) if total_before else 0.0,
        "candidates_count": len(candidates_df),
    }

    cols = ["article_id", "title", "content", "published_at", "source", "url", "category_gold"]
    df_out = df_final[cols].sort_values("article_id").reset_index(drop=True)
    OUTPUT_DEDUP.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_DEDUP, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu: %s (%d dòng)", OUTPUT_DEDUP.relative_to(ROOT), len(df_out))

    removed_df = pd.DataFrame(all_removed)
    if removed_df.empty:
        removed_df = pd.DataFrame(columns=["kept_article_id", "removed_article_id", "duplicate_type", "evidence"])
    removed_df = removed_df.sort_values(["duplicate_type", "kept_article_id", "removed_article_id"]).reset_index(drop=True)
    OUTPUT_REMOVED.parent.mkdir(parents=True, exist_ok=True)
    removed_df.to_csv(OUTPUT_REMOVED, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu: %s (%d dòng)", OUTPUT_REMOVED.relative_to(ROOT), len(removed_df))

    cand_out = candidates_df.sort_values(
        ["decision", "content_similarity", "article_id_1", "article_id_2"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    OUTPUT_CAND.parent.mkdir(parents=True, exist_ok=True)
    cand_out.to_csv(OUTPUT_CAND, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu: %s (%d cặp)", OUTPUT_CAND.relative_to(ROOT), len(cand_out))

    ba_df = compute_before_after(df, df_out)
    OUTPUT_BA.parent.mkdir(parents=True, exist_ok=True)
    ba_df.to_csv(OUTPUT_BA, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu: %s", OUTPUT_BA.relative_to(ROOT))

    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline khử trùng lặp dữ liệu (Task D02)")
    p.add_argument("--input", type=Path, default=INPUT_CSV, help=f"File CSV đầu vào (mặc định: {INPUT_CSV.name})")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        run_pipeline(input_path=args.input)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Pipeline thất bại: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
