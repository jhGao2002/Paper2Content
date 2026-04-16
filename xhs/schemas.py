from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class XHSImagePlan:
    main_subject: str = ""
    scene: str = ""
    pain_point_visual: str = ""
    solution_visual: str = ""
    props: list[str] = field(default_factory=list)
    composition: str = ""
    color_palette: str = ""
    style_keywords: list[str] = field(default_factory=list)
    avoid_elements: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "XHSImagePlan":
        data = payload or {}
        return cls(
            main_subject=str(data.get("main_subject", "")).strip(),
            scene=str(data.get("scene", "")).strip(),
            pain_point_visual=str(data.get("pain_point_visual", "")).strip(),
            solution_visual=str(data.get("solution_visual", "")).strip(),
            props=[str(item).strip() for item in data.get("props", []) if str(item).strip()],
            composition=str(data.get("composition", "")).strip(),
            color_palette=str(data.get("color_palette", "")).strip(),
            style_keywords=[str(item).strip() for item in data.get("style_keywords", []) if str(item).strip()],
            avoid_elements=[str(item).strip() for item in data.get("avoid_elements", []) if str(item).strip()],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class XHSNoteDraft:
    title: str
    audience: str
    core_problem: str
    solved_problem: str
    cover_hook: str
    summary: str
    body: str
    cta: str
    hashtags: list[str] = field(default_factory=list)
    qa_pairs: list[dict[str, str]] = field(default_factory=list)
    image_plan: XHSImagePlan = field(default_factory=XHSImagePlan)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "XHSNoteDraft":
        qa_pairs: list[dict[str, str]] = []
        for item in payload.get("qa_pairs", []):
            if not isinstance(item, dict):
                continue
            qa_pairs.append(
                {
                    "question": str(item.get("question", "")).strip(),
                    "answer": str(item.get("answer", "")).strip(),
                    "takeaway": str(item.get("takeaway", "")).strip(),
                }
            )

        hashtags = [str(tag).strip() for tag in payload.get("hashtags", []) if str(tag).strip()]
        return cls(
            title=str(payload.get("title", "")).strip(),
            audience=str(payload.get("audience", "")).strip(),
            core_problem=str(payload.get("core_problem", "")).strip(),
            solved_problem=str(payload.get("solved_problem", "")).strip(),
            cover_hook=str(payload.get("cover_hook", "")).strip(),
            summary=str(payload.get("summary", "")).strip(),
            body=str(payload.get("body", "")).strip(),
            cta=str(payload.get("cta", "")).strip(),
            hashtags=hashtags,
            qa_pairs=qa_pairs,
            image_plan=XHSImagePlan.from_dict(payload.get("image_plan")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class XHSMCPPublishArgs:
    title: str
    content: str
    images: list[str]
    tags: list[str] = field(default_factory=list)
    is_original: bool = True
    visibility: str = "公开可见"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class XHSNoteArtifact:
    note: XHSNoteDraft
    image_prompt: str
    image_negative_prompt: str
    cover_source_materials: list[dict[str, str]] = field(default_factory=list)
    cover_core_insight: str = ""
    image_paths: list[str] = field(default_factory=list)
    image_error: str = ""
    mcp_args: XHSMCPPublishArgs | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["note"] = self.note.to_dict()
        payload["mcp_args"] = self.mcp_args.to_dict() if self.mcp_args else None
        return payload
