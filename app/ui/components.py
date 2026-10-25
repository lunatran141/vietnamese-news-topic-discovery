"""Thành phần giao diện tái sử dụng: header, KPI, pill, ghi chú, bảng."""

from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st
import json
from pathlib import Path

_GENERIC_NAME = re.compile(r"^(topic|chủ đề)\s*\d+$", re.IGNORECASE)
_NUMERIC_PREFIX = re.compile(r"^\d+_")


def fmt_number(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.3f}"


def fmt_percent(value: float | None) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value:.1%}"


def fmt_seconds(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{fmt_number(value)} giây"


def clean_topic_name(raw: str | None, keywords: str | None = None) -> str:
    """Làm sạch tên topic chỉ ở lớp hiển thị; artifact gốc không đổi."""
    text = "" if raw is None or pd.isna(raw) else str(raw).strip()
    if text.lower() == "outlier":
        return "Outlier"
    if not text or _GENERIC_NAME.match(text):
        words = [w.strip().replace("_", " ") for w in str(keywords or "").split(",") if w.strip()]
        return " · ".join(words[:3]) if words else "Chưa có tên"
    text = _NUMERIC_PREFIX.sub("", text).replace("_", " ")
    return text.replace(" / ", " · ")


def build_topic_labels(bundle, run_id: str) -> dict[int, str]:
    """Lấy tên topic từ mapping của từng mô hình, không sửa artifact gốc."""
    saved_mapping: dict[str, dict] = {}

    root = Path(__file__).resolve().parents[2]

    if run_id == "minilm_title_content_eom_15_5":
        path = (
            root
            / "artifacts/common/hdbscan"
            / run_id
            / "topic_display_mapping.json"
        )

        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8-sig"))

            if payload.get("run_id") != run_id:
                raise ValueError("Mapping HDBSCAN sai run_id")

            saved_mapping = payload["mapping"]

    elif run_id == "lda_k15_seed42":
        path = (
            root
            / "artifacts/common/lda"
            / run_id
            / "topic_display_mapping.json"
        )

        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy mapping LDA: {path}")

        payload = json.loads(path.read_text(encoding="utf-8-sig"))

        if payload.get("run_id") != run_id:
            raise ValueError("Mapping LDA sai run_id")

        if payload.get("schema_version") != "1.0":
            raise ValueError("Mapping LDA sai schema_version")

        entries = payload.get("mapping")

        if not isinstance(entries, dict):
            raise ValueError("Mapping LDA không hợp lệ")

        if set(entries) != {str(i) for i in range(15)}:
            raise ValueError("Mapping LDA phải chứa đúng 15 topic")

        for topic_id, entry in entries.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Topic {topic_id} thiếu dữ liệu mapping")

            label = entry.get("display_name")

            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"Topic {topic_id} thiếu display_name")

        saved_mapping = entries

    session_mapping = st.session_state.get(f"mapping_{run_id}", {})

    if run_id == "lda_k15_seed42":
        mapping = saved_mapping
    else:
        mapping = {
            **saved_mapping,
            **session_mapping,
        }

    labels: dict[int, str] = {}

    for row in bundle.topic_summary.to_dict("records"):
        topic_id = int(row["topic_id"])

        if topic_id == -1:
            labels[topic_id] = "Outlier"
            continue

        custom = mapping.get(str(topic_id), {}).get("display_name")

        name = custom or clean_topic_name(
            row.get("topic_name"),
            row.get("top_keywords"),
        )

        labels[topic_id] = f"T{topic_id} · {name}"

    return labels


def label_of(topic_id: int, labels: dict[int, str]) -> str:
    topic_id = int(topic_id)
    if topic_id == -1:
        return "Outlier"
    return labels.get(topic_id, f"T{topic_id}")


def page_header(
    title: str,
    subtitle: str | None = None,
    chips: dict[str, str] | None = None,
    eyebrow: str | None = None,
) -> None:
    parts = ['<div class="ui-header">']
    if eyebrow:
        parts.append(f'<div class="ui-eyebrow">{html.escape(eyebrow)}</div>')
    parts.append(f'<div class="ui-title">{html.escape(title)}</div>')
    if subtitle:
        parts.append(f'<div class="ui-subtitle">{html.escape(subtitle)}</div>')
    if chips:
        chip_html = "".join(
            f'<span class="ui-chip"><b>{html.escape(k)}</b>{html.escape(str(v))}</span>'
            for k, v in chips.items()
        )
        parts.append(f'<div class="ui-context">{chip_html}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """Mỗi phần tử là (nhãn, giá trị, chú thích hoặc None)."""
    for column, (label, value, hint) in zip(st.columns(len(items)), items):
        hint_html = f'<div class="kpi-hint">{html.escape(hint)}</div>' if hint else ""
        column.markdown(
            f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(str(value))}</div>{hint_html}</div>',
            unsafe_allow_html=True,
        )


def keyword_pills(keywords: list[str]) -> None:
    if not keywords:
        st.caption("Chưa có từ khóa.")
        return
    st.markdown(
        "".join(f'<span class="pill">{html.escape(str(k))}</span>' for k in keywords),
        unsafe_allow_html=True,
    )


def key_value_panel(rows: list[tuple[str, str]]) -> None:
    body = "".join(
        f'<div class="k">{html.escape(k)}</div><div class="v">{html.escape(str(v))}</div>'
        for k, v in rows
    )
    st.markdown(f'<div class="kv">{body}</div>', unsafe_allow_html=True)


def note(text: str, kind: str = "info") -> None:
    css = "note warn" if kind == "warn" else "note"
    st.markdown(f'<div class="{css}">{html.escape(text)}</div>', unsafe_allow_html=True)


def doc_row(title: str, meta: str) -> None:
    st.markdown(
        f'<div class="doc-row"><div class="doc-title">{html.escape(title)}</div>'
        f'<div class="doc-meta">{html.escape(meta)}</div></div>',
        unsafe_allow_html=True,
    )


def data_table(frame: pd.DataFrame, column_config: dict | None = None, height: int | None = None) -> None:
    options = {"height": height} if height is not None else {}
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config=column_config,
        **options,
    )


def table_download(frame: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8-sig"),
        filename,
        mime="text/csv",
    )
