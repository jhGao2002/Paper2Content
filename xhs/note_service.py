from __future__ import annotations

import json
import re
import time
from pathlib import Path

from xhs.image_service import build_cover_negative_prompt, build_cover_prompt, generate_cover_images
from xhs.publish_service import build_mcp_publish_args, publish_note_via_mcp_sync
from xhs.schemas import XHSNoteArtifact, XHSNoteDraft


def _log_progress(message: str) -> None:
    print(f"[XHSNoteService] {message}", flush=True)


def _conversation_to_text(history: list[dict]) -> str:
    lines: list[str] = []
    for index, item in enumerate(history, start=1):
        role = "用户" if item.get("role") == "user" else "助手"
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"[{index}] {role}：{content}")
    return "\n".join(lines)


def _split_sentences(text: str) -> list[str]:
    parts = [item.strip() for item in re.split(r"[.!?。！？]\s*", text.replace("\n", " ")) if item.strip()]
    if parts:
        return parts
    return [item.strip() for item in text.splitlines() if item.strip()]


def _fallback_cover_brief_from_source(source_materials: list[dict[str, str]]) -> dict[str, object]:
    if not source_materials:
        return {"core_insight": "", "supporting_elements": []}

    excerpt = str(source_materials[0].get("excerpt", "")).strip()
    title = str(source_materials[0].get("title", "")).strip().lower()
    sentences = _split_sentences(excerpt)
    core_insight = ". ".join(sentences[:2]).strip()
    if core_insight and excerpt.count(".") > 0 and not core_insight.endswith("."):
        core_insight += "."

    text = f"{title}\n{excerpt}".lower()
    elements: list[str] = []
    keyword_map = [
        ("reference", "reference image panel"),
        ("style image", "style reference image"),
        ("retrieval", "retrieved image grid"),
        ("attention", "attention heatmap overlay"),
        ("diffusion", "diffusion denoising trajectory"),
        ("style transfer", "content image and stylized result"),
        ("memory", "memory cards or note blocks"),
    ]
    for keyword, visual in keyword_map:
        if keyword in text and visual not in elements:
            elements.append(visual)

    if not elements:
        elements = ["paper page highlights", "diagram card", "before-and-after comparison"]

    return {
        "core_insight": core_insight or excerpt[:220].strip(),
        "supporting_elements": elements[:4],
        "main_subject": "paper mechanism illustration with result comparison",
        "scene": "editorial science cover scene focused on the paper's method and outcome",
        "composition": "central mechanism with supporting visual elements around it",
        "pain_point_visual": "the paper's original problem setting, input condition, or challenge",
        "solution_visual": "the proposed method, key mechanism, or improved output result",
        "style_keywords": ["editorial science cover", "clean visual explanation", "high clarity"],
    }


def _extract_json_payload(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[-1]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"模型未返回合法 JSON：{raw}")
        return json.loads(text[start : end + 1])


def _fallback_note(history: list[dict]) -> XHSNoteDraft:
    preview = _conversation_to_text(history)
    summary = preview[:180] + ("..." if len(preview) > 180 else "")
    return XHSNoteDraft(
        title="这次问答整理",
        audience="知识型内容创作者",
        core_problem="对话里提到的问题还没有被整理成可复用笔记",
        solved_problem="把零散 QA 收敛成可发布的知识分享",
        cover_hook="把问答变成一篇清晰的图文笔记",
        summary=summary or "基于当前对话整理出的笔记草稿。",
        body=(
            "我把这次对话里最重要的问答先收拢成一篇笔记草稿。\n\n"
            "如果你也经常在聊天里得到很多有用信息，但最后没有沉淀下来，这种整理方式会更容易复用。\n\n"
            "建议你回看对话，把最关键的问题、答案和可执行动作各挑 1-3 条，再补成一篇可分享内容。"
        ),
        cta="如果你也想把聊天记录沉淀成内容，可以继续补充更完整的问答上下文。",
        hashtags=["知识管理", "笔记整理", "效率工具"],
    )


def _repair_note(note: XHSNoteDraft) -> XHSNoteDraft:
    note.title = note.title[:20] or "问答整理笔记"
    if not note.audience:
        note.audience = "知识型内容读者"
    if not note.core_problem:
        note.core_problem = "用户在对话中遇到的核心问题"
    if not note.solved_problem:
        note.solved_problem = note.core_problem
    if not note.cover_hook:
        note.cover_hook = note.solved_problem
    if not note.summary:
        note.summary = note.solved_problem
    if not note.body:
        note.body = f"{note.summary}\n\n核心问题：{note.core_problem}\n解决结果：{note.solved_problem}"
    if not note.cta:
        note.cta = "如果你也在处理类似问题，欢迎按这套方式继续拆解。"
    if not note.hashtags:
        note.hashtags = ["知识分享", "经验总结"]
    if not note.image_plan.main_subject:
        note.image_plan.main_subject = "一位正在整理知识点的创作者"
    if not note.image_plan.scene:
        note.image_plan.scene = "桌面工作区，电脑与纸质笔记同时出现"
    if not note.image_plan.pain_point_visual:
        note.image_plan.pain_point_visual = f"和“{note.core_problem}”相关的混乱、卡住或反复试错"
    if not note.image_plan.solution_visual:
        note.image_plan.solution_visual = f"和“{note.solved_problem}”相关的清晰、有秩序、被解决后的状态"
    return note


class XHSNoteService:
    def __init__(self, llm, cover_source_provider=None):
        self.llm = llm
        self.cover_source_provider = cover_source_provider

    def _build_generation_prompt(self, history: list[dict]) -> str:
        conversation = _conversation_to_text(history)
        return (
            "你是一名擅长把问答记录整理成小红书图文笔记的内容编辑。\n"
            "请严格根据给定对话整理，不要补充对话中没有出现的结论，不要引用不存在的数据。\n"
            "输出必须是 JSON，不要附加解释，不要使用 Markdown 代码块。\n\n"
            "JSON 结构如下：\n"
            "{\n"
            '  "title": "20字以内的标题",\n'
            '  "audience": "目标读者",\n'
            '  "core_problem": "对话里用户真正遇到的问题",\n'
            '  "solved_problem": "这篇内容最终解决了什么问题",\n'
            '  "cover_hook": "适合封面表达的一句话，不要口号化",\n'
            '  "summary": "80字以内摘要",\n'
            '  "body": "适合小红书图文正文的纯文本，多段换行，450-700字，不要输出hashtags列表，不要虚构额外案例",\n'
            '  "cta": "结尾互动引导，1句",\n'
            '  "hashtags": ["3-6个标签，不要带#"],\n'
            '  "qa_pairs": [\n'
            '    {"question": "关键问题1", "answer": "对应答案1", "takeaway": "一句提炼"},\n'
            '    {"question": "关键问题2", "answer": "对应答案2", "takeaway": "一句提炼"}\n'
            "  ],\n"
            '  "image_plan": {\n'
            '    "main_subject": "画面主体",\n'
            '    "scene": "真实可画的场景",\n'
            '    "pain_point_visual": "问题阶段的视觉表现",\n'
            '    "solution_visual": "解决后的视觉表现",\n'
            '    "props": ["最多4个辅助物件"],\n'
            '    "composition": "构图方式",\n'
            '    "color_palette": "颜色方案",\n'
            '    "style_keywords": ["最多4个风格关键词"],\n'
            '    "avoid_elements": ["不希望出现的元素"]\n'
            "  }\n"
            "}\n\n"
            "封面设计要求：\n"
            "1. 重点表现“问题被解决了什么”，用场景、人物动作、物件状态、前后对比来表达。\n"
            "2. 不要把画面设计成纯文字海报，不要依赖大段中文。\n"
            "3. 画面要适合后续做统一风格迁移，所以主体、场景、物件要清楚，不要过度复杂。\n"
            "4. image_plan 要尽量具体，方便直接喂给文生图模型。\n\n"
            f"对话记录：\n{conversation}"
        )

    def generate_note(self, history: list[dict]) -> XHSNoteDraft:
        if not history:
            raise ValueError("当前会话还没有可整理的问答记录。")

        _log_progress(f"开始整理笔记正文，history条数={len(history)}")
        prompt = self._build_generation_prompt(history)
        try:
            llm_start = time.perf_counter()
            _log_progress("调用 LLM 生成结构化笔记 JSON")
            raw = self.llm.invoke(prompt).content
            note = XHSNoteDraft.from_dict(_extract_json_payload(str(raw)))
            _log_progress(f"LLM 笔记生成完成，耗时={time.perf_counter() - llm_start:.2f}s")
        except Exception:
            _log_progress("LLM 笔记生成失败，回退到规则草稿")
            note = _fallback_note(history)
        repaired = _repair_note(note)
        _log_progress(f"笔记草稿就绪，标题={repaired.title}")
        return repaired

    def _get_cover_source_materials(self) -> list[dict[str, str]]:
        if self.cover_source_provider is None:
            return []
        try:
            materials = self.cover_source_provider() or []
        except Exception as exc:
            _log_progress(f"获取封面源材料失败：{exc}")
            return []
        normalized: list[dict[str, str]] = []
        for item in materials:
            if not isinstance(item, dict):
                continue
            excerpt = str(item.get("excerpt", "")).strip()
            if not excerpt:
                continue
            normalized.append(
                {
                    "source": str(item.get("source", "")).strip(),
                    "title": str(item.get("title", "")).strip(),
                    "excerpt": excerpt,
                }
            )
        return normalized

    def _build_cover_brief(
        self,
        note: XHSNoteDraft,
        source_materials: list[dict[str, str]],
    ) -> dict[str, object]:
        if not source_materials:
            return {
                "core_insight": note.solved_problem or note.cover_hook or note.summary,
                "supporting_elements": list(note.image_plan.props),
            }

        source_text = "\n\n".join(
            f"[{item['title'] or item['source']}]\n{item['excerpt']}" for item in source_materials
        )
        prompt = (
            "你是一名负责论文封面视觉策划的编辑。\n"
            "请只根据给定论文原文的 abstract 或 introduction 片段，提炼适合文生图封面的核心洞察。\n"
            "不要使用对话总结，不要补充原文之外的结论。\n"
            "如果原文是英文，core_insight 必须保持英文；如果原文是中文，保持中文。\n"
            "输出 JSON，不要解释：\n"
            "{\n"
            '  "core_insight": "1-2句，保留原语言，表达论文最核心洞察",\n'
            '  "supporting_elements": ["2-5个辅助理解该洞察的视觉元素，不要只是重复文字"],\n'
            '  "pain_point_visual": "研究对象、输入条件或待解决问题的视觉表现",\n'
            '  "solution_visual": "关键机制、效果或输出结果的视觉表现",\n'
            '  "main_subject": "画面主体",\n'
            '  "scene": "适合封面的场景",\n'
            '  "composition": "构图",\n'
            '  "style_keywords": ["最多4个风格关键词"]\n'
            "}\n\n"
            f"论文原文片段：\n{source_text}"
        )

        try:
            _log_progress("调用 LLM 从原文摘要/引言提炼封面核心洞察")
            raw = self.llm.invoke(prompt).content
            data = _extract_json_payload(str(raw))
        except Exception as exc:
            _log_progress(f"原文洞察提炼失败，回退到原文片段启发式提炼：{exc}")
            fallback = _fallback_cover_brief_from_source(source_materials)
            if fallback.get("core_insight"):
                note.image_plan.main_subject = str(fallback.get("main_subject", "")).strip() or note.image_plan.main_subject
                note.image_plan.scene = str(fallback.get("scene", "")).strip() or note.image_plan.scene
                note.image_plan.composition = str(fallback.get("composition", "")).strip() or note.image_plan.composition
                note.image_plan.pain_point_visual = (
                    str(fallback.get("pain_point_visual", "")).strip() or note.image_plan.pain_point_visual
                )
                note.image_plan.solution_visual = (
                    str(fallback.get("solution_visual", "")).strip() or note.image_plan.solution_visual
                )
                fallback_styles = [str(item).strip() for item in fallback.get("style_keywords", []) if str(item).strip()]
                if fallback_styles:
                    note.image_plan.style_keywords = fallback_styles
                return fallback
            return {
                "core_insight": note.solved_problem or note.cover_hook or note.summary,
                "supporting_elements": list(note.image_plan.props),
            }

        core_insight = str(data.get("core_insight", "")).strip() or note.solved_problem or note.summary
        supporting = [str(item).strip() for item in data.get("supporting_elements", []) if str(item).strip()]
        note.image_plan.main_subject = str(data.get("main_subject", "")).strip() or note.image_plan.main_subject
        note.image_plan.scene = str(data.get("scene", "")).strip() or note.image_plan.scene
        note.image_plan.composition = str(data.get("composition", "")).strip() or note.image_plan.composition
        note.image_plan.pain_point_visual = (
            str(data.get("pain_point_visual", "")).strip() or note.image_plan.pain_point_visual
        )
        note.image_plan.solution_visual = (
            str(data.get("solution_visual", "")).strip() or note.image_plan.solution_visual
        )
        extracted_styles = [str(item).strip() for item in data.get("style_keywords", []) if str(item).strip()]
        if extracted_styles:
            note.image_plan.style_keywords = extracted_styles
        return {
            "core_insight": core_insight,
            "supporting_elements": supporting,
        }

    def generate_note_artifact(
        self,
        history: list[dict],
        generate_images: bool = False,
        image_count: int = 1,
        output_dir: str | None = None,
        is_original: bool = True,
        visibility: str = "公开可见",
    ) -> dict:
        overall_start = time.perf_counter()
        _log_progress("进入 generate_note_artifact")
        note = self.generate_note(history)
        source_materials = self._get_cover_source_materials()
        if source_materials:
            _log_progress(f"找到 {len(source_materials)} 份封面源材料，优先使用原文摘要/引言")
        else:
            _log_progress("未找到原文摘要/引言源材料，回退使用笔记结论")
        cover_brief = self._build_cover_brief(note, source_materials)
        _log_progress("开始生成封面 prompt")
        image_prompt = build_cover_prompt(
            note,
            cover_core_insight=str(cover_brief.get("core_insight", "")).strip(),
            supporting_elements=list(cover_brief.get("supporting_elements", []) or []),
        )
        negative_prompt = build_cover_negative_prompt(note)
        _log_progress(f"封面 prompt 已生成，长度={len(image_prompt)}")
        target_dir = Path(output_dir) if output_dir else None
        image_paths: list[str] = []
        image_error = ""

        if generate_images:
            try:
                _log_progress("开始调用图片生成服务")
                image_paths = generate_cover_images(
                    note,
                    image_count=image_count,
                    output_dir=target_dir,
                    cover_core_insight=str(cover_brief.get("core_insight", "")).strip(),
                    supporting_elements=list(cover_brief.get("supporting_elements", []) or []),
                )
                _log_progress(f"图片生成完成，数量={len(image_paths)}")
            except Exception as exc:
                image_error = str(exc)
                _log_progress(f"图片生成失败：{image_error}")

        artifact = XHSNoteArtifact(
            note=note,
            image_prompt=image_prompt,
            image_negative_prompt=negative_prompt,
            cover_source_materials=source_materials,
            cover_core_insight=str(cover_brief.get("core_insight", "")).strip(),
            image_paths=image_paths,
            image_error=image_error,
            mcp_args=build_mcp_publish_args(
                note=note,
                image_paths=image_paths,
                is_original=is_original,
                visibility=visibility,
            ),
        )
        _log_progress(f"artifact 构建完成，总耗时={time.perf_counter() - overall_start:.2f}s")
        return artifact.to_dict()

    def publish_generated_note(self, artifact: dict) -> dict:
        _log_progress("开始组装发布参数")
        mcp_args = artifact.get("mcp_args") or {}
        note = artifact.get("note") or {}
        publish_args = build_mcp_publish_args(
            note=XHSNoteDraft.from_dict(note),
            image_paths=list(mcp_args.get("images", [])),
            is_original=bool(mcp_args.get("is_original", True)),
            visibility=str(mcp_args.get("visibility", "公开可见")),
        )
        _log_progress("开始调用小红书 MCP 发布")
        return publish_note_via_mcp_sync(publish_args)
