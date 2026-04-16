from __future__ import annotations

import json
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
    def __init__(self, llm):
        self.llm = llm

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
        _log_progress("开始生成封面 prompt")
        image_prompt = build_cover_prompt(note)
        negative_prompt = build_cover_negative_prompt(note)
        _log_progress(f"封面 prompt 已生成，长度={len(image_prompt)}")
        target_dir = Path(output_dir) if output_dir else None
        image_paths: list[str] = []
        image_error = ""

        if generate_images:
            try:
                _log_progress("开始调用图片生成服务")
                image_paths = generate_cover_images(note, image_count=image_count, output_dir=target_dir)
                _log_progress(f"图片生成完成，数量={len(image_paths)}")
            except Exception as exc:
                image_error = str(exc)
                _log_progress(f"图片生成失败：{image_error}")

        artifact = XHSNoteArtifact(
            note=note,
            image_prompt=image_prompt,
            image_negative_prompt=negative_prompt,
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
