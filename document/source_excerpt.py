from __future__ import annotations

import re

from document.registry import get_document


def _normalize_heading(line: str) -> str:
    return line.strip().lstrip("#").strip().lower()


def _is_section_boundary(line: str) -> bool:
    stripped = line.strip()
    normalized = _normalize_heading(stripped)
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if re.match(r"^\d+(\.\d+)*\s+[A-Za-z].*", stripped):
        return True
    if normalized.startswith(("keywords", "index terms", "references", "related work", "method", "methods", "experiments")):
        return True
    if re.match(r"^[一二三四五六七八九十]+[、.．]\s*", stripped):
        return True
    if normalized.startswith(("关键词", "相关工作", "方法", "实验", "参考文献")):
        return True
    return False


def _collect_section(lines: list[str], start: int, max_chars: int = 1200) -> str:
    parts: list[str] = []
    total = 0
    for idx in range(start + 1, len(lines)):
        line = lines[idx].strip()
        if not line:
            if parts:
                parts.append("")
            continue
        if _is_section_boundary(line):
            break
        total += len(line)
        parts.append(line)
        if total >= max_chars:
            break
    text = "\n".join(parts).strip()
    return text[:max_chars].strip()


def _find_section(lines: list[str], candidates: tuple[str, ...]) -> str:
    for idx, line in enumerate(lines):
        normalized = _normalize_heading(line)
        if normalized in candidates:
            section = _collect_section(lines, idx)
            if section:
                return section
        for candidate in candidates:
            if normalized.startswith(candidate + " "):
                section = _collect_section(lines, idx)
                if section:
                    return section
    return ""


def _select_source_excerpt_from_docs(docs: list) -> str:
    parent_texts: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        text = str(doc.metadata.get("parent_text") or doc.page_content or "").strip()
        if text and text not in seen:
            parent_texts.append(text)
            seen.add(text)

    if not parent_texts:
        return ""

    combined = "\n\n".join(parent_texts[:6])
    lines = [line for line in combined.splitlines()]

    abstract = _find_section(lines, ("abstract", "摘要"))
    if abstract:
        return abstract

    introduction = _find_section(lines, ("introduction", "引言"))
    if introduction:
        return introduction

    return parent_texts[0][:1200].strip()


def collect_cover_source_materials(pdf_store, source_names: list[str], user_id: str = "") -> list[dict]:
    target_sources = list(source_names)
    if not target_sources:
        target_sources = pdf_store.list_sources()[:3]

    materials: list[dict] = []
    for source in target_sources:
        must_filters = [{"key": "source", "match": {"value": source}}]
        if user_id:
            must_filters.append({"key": "user_id", "match": {"value": user_id}})
        docs = pdf_store.get_documents(filter={"must": must_filters}, limit=80)
        excerpt = _select_source_excerpt_from_docs(docs)
        meta = get_document(source) or {}
        title = str(meta.get("title", source)).strip() or source
        summary = str(meta.get("summary", "")).strip()
        if not excerpt and summary:
            excerpt = summary
        if excerpt:
            materials.append(
                {
                    "source": source,
                    "title": title,
                    "excerpt": excerpt,
                }
            )
    return materials
