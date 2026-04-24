from __future__ import annotations

import json
import re
import time
from pathlib import Path

from xhs.image_service import build_cover_negative_prompt, generate_cover_images
from xhs.publish_service import build_mcp_publish_args_from_payload, publish_note_via_mcp_sync
from xhs.style_transfer_service import style_transfer_sync
from xhs.schemas import XHSNoteArtifact, XHSNoteDraft, XHSPreparedUploadPayload


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


def _topic_summary(text: str, limit: int = 28) -> str:
    clean = re.sub(r"\s+", " ", str(text or "").strip())
    if not clean:
        return "未命名主题"
    return clean[:limit] + ("..." if len(clean) > limit else "")


def _parse_selection_numbers(text: str) -> list[int]:
    return list(dict.fromkeys(int(item) for item in re.findall(r"\d+", text)))


def _truncate_text(text: str, limit: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _fallback_short_title(text: str, limit: int = 15) -> str:
    clean = re.sub(r"\s+", " ", str(text or "").strip())
    if not clean:
        return ""

    clean = re.sub(r"[“”\"'`]+", "", clean)
    clean = re.sub(r"[，。！？；：,.!?;:]+$", "", clean)
    clean = re.sub(
        r"^(例如|比如|像是|关于|对于|如果|当|在|做|讲|说|聊聊|我想问|我想写|我想发|帮我写|帮我总结)",
        "",
        clean,
    ).strip()

    preferred_parts = [item.strip(" ，。！？；：,.!?;:") for item in _split_sentences(clean) if item.strip()]
    if preferred_parts:
        clean = preferred_parts[0]

    for splitter in ("可以", "可能", "的时候", "时", "如何", "怎么", "为什么"):
        if splitter in clean:
            prefix = clean.split(splitter, 1)[0].strip(" ，。！？；：,.!?;:")
            if len(prefix) >= 4:
                clean = prefix
                break

    return _truncate_text(clean, limit)


def _looks_like_workflow_message(text: str) -> bool:
    raw = re.sub(r"\s+", " ", str(text or "").strip()).lower()
    if not raw:
        return True
    if re.fullmatch(r"[\d,\s，、]+", raw):
        return True

    workflow_keywords = (
        "发布到小红书",
        "发布笔记",
        "发布图文笔记",
        "小红书发布",
        "确认发布",
        "开始发布",
        "进入下一步",
        "继续下一步",
        "继续做封面",
        "开始做封面",
        "满意",
        "没问题",
        "确认内容",
        "内容满意",
        "需要风格迁移",
        "不需要风格迁移",
        "确认发布",
        "取消发布",
    )
    return any(keyword in raw for keyword in workflow_keywords)


def _looks_like_publishable_question(text: str) -> bool:
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if len(raw) < 4:
        return False
    if _looks_like_workflow_message(raw):
        return False

    question_keywords = (
        "?",
        "？",
        "什么",
        "哪些",
        "哪个",
        "怎么",
        "如何",
        "为什么",
        "为何",
        "是否",
        "能否",
        "有没有",
        "吗",
        "么",
        "呢",
        "区别",
        "差异",
        "对比",
        "创新点",
        "优缺点",
        "原理",
        "流程",
        "作用",
        "含义",
    )
    if any(keyword in raw for keyword in question_keywords):
        return True

    ask_prefixes = (
        "请总结",
        "请解释",
        "请分析",
        "请对比",
        "请说明",
        "帮我总结",
        "帮我解释",
        "帮我分析",
        "帮我对比",
        "详细说说",
        "展开讲讲",
    )
    return any(raw.startswith(prefix) for prefix in ask_prefixes)


def _extract_hash_tags(text: str) -> list[str]:
    return list(dict.fromkeys(tag.strip() for tag in re.findall(r"#([^\s#]+)", text) if tag.strip()))


def _normalize_visibility(text: str, default: str = "公开可见") -> str:
    raw = str(text or "")
    if "仅自己" in raw or "自己可见" in raw or "私密" in raw:
        return "仅自己可见"
    if "好友" in raw:
        return "好友可见"
    return default


def _is_confirm_message(text: str) -> bool:
    raw = str(text or "").strip()
    keywords = ("确认发布", "可以发布", "确认", "发布吧", "发吧", "开始发布", "确定发布")
    return any(keyword in raw for keyword in keywords)


def _is_cancel_message(text: str) -> bool:
    raw = str(text or "").strip()
    keywords = ("取消发布", "先别发", "不要发布", "不发布了", "停止发布")
    return any(keyword in raw for keyword in keywords)


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


def _build_upload_content(note: XHSNoteDraft) -> str:
    content = note.body.strip()
    if note.cta and note.cta not in content:
        content = f"{content}\n\n{note.cta}".strip()
    return content



class XHSNoteService:
    def __init__(self, llm, fast_llm=None):
        self.llm = llm
        self.fast_llm = fast_llm or llm

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

        _log_progress(f"Building note draft from history, turns={len(history)}")
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

    def build_publish_candidates(
        self,
        history: list[dict],
        memory_notes: list[str] | None = None,
    ) -> list[dict[str, str]]:
        candidates: list[dict[str, str]] = []
        seen: set[str] = set()

        pending_user = ""
        for item in history:
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                pending_user = content
                continue
            if role != "assistant" or not pending_user:
                continue
            if not _looks_like_publishable_question(pending_user):
                pending_user = ""
                continue
            if any(keyword in pending_user for keyword in ("发布到小红书", "发布笔记", "发布图文笔记", "小红书发布")):
                pending_user = ""
                continue
            if "是否确认发布" in content or "请回复编号" in content:
                pending_user = ""
                continue
            normalized = f"qa::{pending_user}::{content}"
            if normalized in seen:
                pending_user = ""
                continue
            seen.add(normalized)
            candidates.append(
                {
                    "id": str(len(candidates) + 1),
                    "type": "qa",
                    "topic": _topic_summary(pending_user),
                    "question": pending_user,
                    "content": content,
                }
            )
            pending_user = ""

        for note in memory_notes or []:
            content = str(note or "").strip()
            if not content:
                continue
            normalized = f"note::{content}"
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(
                {
                    "id": str(len(candidates) + 1),
                    "type": "note",
                    "topic": _topic_summary(content),
                    "question": "",
                    "content": content,
                }
            )
        return candidates

    def render_candidate_list(self, candidates: list[dict[str, str]]) -> str:
        if not candidates:
            return (
                "当前会话里还没有可用于发布的小红书素材。\n"
                "你可以先继续提问、让系统回答，或者先让我帮你整理/保存笔记，再回来发小红书。"
            )

        lines = ["我先帮你整理出当前会话里可直接拿来发布的素材，请选择编号，可多选："]
        for index, item in enumerate(candidates, start=1):
            label = item.get("question") or item.get("topic") or "未命名主题"
            kind = "问答" if item.get("type") == "qa" else "笔记"
            lines.append(f"{index}. [{kind}] {label}")
        lines.append("请直接回复编号，例如：`1,3` 或 `2 4 5`。")
        return "\n".join(lines)

    def build_publish_note_from_candidates(self, selected_items: list[dict[str, str]]) -> tuple[XHSNoteDraft, str]:
        history: list[dict] = []
        for item in selected_items:
            question = str(item.get("question", "")).strip()
            content = str(item.get("content", "")).strip()
            if item.get("type") == "qa" and question:
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": content})
            else:
                history.append({"role": "assistant", "content": content})

        note = self.generate_note(history)
        article_title = note.title.strip() or "Default Topic"
        note.body = self._finalize_publish_body(note.body, article_title)
        note.summary = _truncate_text(note.summary, 80)
        note.title = _truncate_text(note.title or _topic_summary(article_title, limit=20), 20)
        return note, article_title

    def _finalize_publish_body(self, body: str, article_title: str) -> str:
        content = str(body or "").strip()
        suffix = f"\n\n文章题目：{article_title.strip() or '本次分享主题'}"
        if suffix.strip() not in content:
            content = f"{content}{suffix}".strip()

        if len(content) <= 1000:
            return content

        prompt = (
            "请在不脱离原始信息的前提下压缩下面这段小红书正文，保留发布风格，"
            "总长度控制在 1000 字以内，结尾必须保留“文章题目：xxx”。"
            "只输出压缩后的正文，不要解释。\n\n"
            f"{content}"
        )
        try:
            compressed = str(self.llm.invoke(prompt).content).strip()
            compressed = compressed or content
        except Exception:
            compressed = content
        if len(compressed) <= 1000:
            return compressed
        suffix = suffix.strip()
        budget = max(0, 1000 - len(suffix) - 2)
        return f"{_truncate_text(compressed, budget)}\n\n{suffix}"

    def build_cover_prompt_from_user_prompt(
        self,
        note: XHSNoteDraft,
        user_prompt: str,
    ) -> str:
        note.title = self.summarize_title_from_prompt(
            user_prompt,
            fallback_title=note.title,
            limit=15,
        ) or note.title
        return self._build_title_based_cover_prompt(note.title)

    def summarize_title_from_prompt(
        self,
        user_prompt: str,
        fallback_title: str = "",
        limit: int = 15,
    ) -> str:
        prompt_text = str(user_prompt or "").strip()
        if not prompt_text:
            return _fallback_short_title(fallback_title, limit=limit)

        prompt = (
            "你是小红书图文标题编辑。请根据用户提供的写作意图，"
            f"总结成一个适合图文笔记发布的中文标题，要求不超过{limit}个字。\n"
            "要求：\n"
            "1. 必须是标题，不要复述整句原话。\n"
            "2. 语言精炼，有传播感，可以偏爆款表达。\n"
            "3. 只输出标题本身，不要解释，不要引号，不要序号。\n\n"
            f"用户prompt：{prompt_text}"
        )
        try:
            raw = str(self.llm.invoke(prompt).content).strip()
        except Exception as exc:
            _log_progress(f"标题总结失败，改用规则兜底：{exc}")
            raw = ""

        candidate = re.sub(r"\s+", "", raw)
        candidate = re.sub(r"^[#\-\d.、\s]+", "", candidate)
        candidate = re.sub(r"[“”\"'`]+", "", candidate)
        candidate = re.sub(r"[，。！？；：,.!?;:]+$", "", candidate)

        if not candidate:
            candidate = _fallback_short_title(prompt_text, limit=limit)
        if not candidate:
            candidate = _fallback_short_title(fallback_title, limit=limit)
        return _truncate_text(candidate, limit)

    def _build_title_based_cover_prompt(self, title: str) -> str:
        clean_title = str(title or "").strip()
        if not clean_title:
            return ""

        prompt = (
            "# Role\n"
            "You are a top-tier visual concept designer and prompt engineer. Your core task is to transform a refined viewpoint title "
            "into a highly striking English VLM prompt, and force the visual model to render the exact original title text as the core visual focus.\n\n"
            "# Objective & Visual Style\n"
            "Create viewpoint-driven conceptual art.\n"
            "1. Use split or opposing composition, such as left-right split, diagonal split, or broken symmetry.\n"
            "2. Use symbolic figures or symbolic objects instead of real faces or literal daily-life scenes.\n"
            "3. Use strong warm-cool contrast like teal and orange or ice blue and neon red.\n"
            "4. Make the central conflict obvious at first glance.\n"
            "5. The exact full title text must be a core visual focal point.\n\n"
            "# Output Format\n"
            "Output one JSON object only.\n"
            "{\n"
            '  "vlm_prompt": "final English prompt"\n'
            "}\n\n"
            "# Prompt Generation Rules\n"
            '- Use wording like `Bold 3D typography saying "原本的标题"` or `The exact text "原本的标题" is written in...`.\n'
            "- Except for the quoted original title, all content must be pure English comma-separated phrases.\n"
            "- The prompt must end with `masterpiece, 8k resolution, highly detailed, dramatic lighting, conceptual art`.\n\n"
            f"Input: {clean_title}"
        )
        try:
            _log_progress("Using primary llm to build the advanced English cover prompt from title")
            raw = str(self.llm.invoke(prompt).content).strip()
            payload = _extract_json_payload(raw)
            candidate = str(payload.get("vlm_prompt", "")).strip()
            if candidate:
                return candidate
        except Exception as exc:
            _log_progress(f"标题驱动高级 prompt 生成失败，回退模板：{exc}")

        return (
            f'A dramatic split composition, symbolic geometric elements, central conflict, '
            f'bold 3D typography saying "{clean_title}", strong teal and orange contrast, '
            f'glowing abstract objects, conceptual editorial illustration, masterpiece, '
            f'8k resolution, highly detailed, dramatic lighting, conceptual art'
        )

    def revise_publish_note(
        self,
        note: XHSNoteDraft,
        article_title: str,
        instruction: str,
    ) -> XHSNoteDraft:
        prompt = (
            "你是小红书内容编辑。请根据用户的修改意见，调整这篇图文草稿。"
            "不要脱离原始内容瞎编。"
            "输出必须是 JSON，不要解释，不要使用 Markdown。\n\n"
            f"当前标题：{note.title}\n"
            f"当前摘要：{note.summary}\n"
            f"当前正文：{note.body}\n"
            f"当前标签：{', '.join(note.hashtags)}\n"
            f"文章题目：{article_title}\n"
            f"修改意见：{instruction}\n\n"
            "输出 JSON：\n"
            "{\n"
            '  "title": "20字以内标题",\n'
            '  "summary": "80字以内摘要",\n'
            '  "body": "修改后的正文，必须适合小红书，后续会自动补文章题目",\n'
            '  "hashtags": ["3-6个标签"]\n'
            "}"
        )
        try:
            raw = self.llm.invoke(prompt).content
            payload = _extract_json_payload(str(raw))
            note.title = _truncate_text(str(payload.get("title", note.title)).strip() or note.title, 20)
            note.summary = _truncate_text(str(payload.get("summary", note.summary)).strip() or note.summary, 80)
            note.body = self._finalize_publish_body(
                str(payload.get("body", note.body)).strip() or note.body,
                article_title,
            )
            hashtags = [str(item).strip() for item in payload.get("hashtags", []) if str(item).strip()]
            if hashtags:
                note.hashtags = hashtags[:6]
        except Exception as exc:
            _log_progress(f"正文修改回退到规则处理：{exc}")
            note.body = self._finalize_publish_body(note.body, article_title)
        return note

    def revise_cover_prompt(
        self,
        note: XHSNoteDraft,
        current_prompt: str,
        instruction: str,
    ) -> str:
        prompt = (
            "你是小红书封面图 prompt 优化师。请基于当前 prompt 和用户修改意见，"
            "输出一段新的、更完整的中文生图 prompt。不能偏离用户原意。"
            "只输出新 prompt，不要解释。\n\n"
            f"当前 prompt：\n{current_prompt}\n\n"
            f"用户修改意见：\n{instruction}\n\n"
            f"当前图文主题：{note.solved_problem or note.cover_hook or note.summary}"
        )
        try:
            updated = str(self.llm.invoke(prompt).content).strip()
            return updated or current_prompt
        except Exception:
            return current_prompt

    def render_publish_confirmation(self, workflow: dict) -> str:
        selected_items = workflow.get("selected_items", []) or []
        note_data = workflow.get("note_draft") or {}
        note = XHSNoteDraft.from_dict(note_data)
        image_prompt = str(workflow.get("image_prompt", "")).strip()
        visibility = str(workflow.get("visibility", "公开可见")).strip() or "公开可见"
        article_title = str(workflow.get("article_title", "")).strip()
        use_style_transfer = bool(workflow.get("use_style_transfer"))
        selected_style_image = str(workflow.get("selected_style_image", "")).strip()
        if use_style_transfer:
            style_summary = "使用远程 MCP 默认风格" if selected_style_image == "__remote_default__" else (selected_style_image or "使用 MCP 服务默认风格图")
        else:
            style_summary = "不做风格迁移"

        lines = ["发布草稿已经准备好了，请你最后确认：", "", "已选素材："]
        for index, item in enumerate(selected_items, start=1):
            label = item.get("question") or item.get("topic") or "未命名主题"
            lines.append(f"{index}. {label}")

        lines.extend(
            [
                "",
                f"发布标题：{note.title}",
                f"文章题目：{article_title or '未设置'}",
                f"标签：{'、'.join(note.hashtags) if note.hashtags else '未设置'}",
                f"可见性：{visibility}",
                "图片：1 张封面图（确认后自动生成）",
                f"风格迁移：{style_summary}",
                "",
                "正文预览：",
                note.body,
                "",
                "封面图高级 prompt：",
                image_prompt or "未生成",
                "",
                "如果你要修改，可以直接说“把正文改短一点”“封面 prompt 更强调方法创新”“改成仅自己可见”等。",
                "如果没问题，请明确回复：`确认发布`。",
            ]
        )
        return "\n".join(lines)

    def build_prepared_payload(
        self,
        note: XHSNoteDraft,
        article_title: str,
        visibility: str = "公开可见",
        image_paths: list[str] | None = None,
        is_original: bool = True,
    ) -> XHSPreparedUploadPayload:
        note.body = self._finalize_publish_body(note.body, article_title)
        return XHSPreparedUploadPayload(
            title=note.title.strip(),
            content=note.body,
            images=list(image_paths or []),
            tags=list(dict.fromkeys(note.hashtags)),
            is_original=is_original,
            visibility=_normalize_visibility(visibility, default="公开可见"),
        )

    def publish_confirmed_workflow(self, workflow: dict) -> dict:
        note = XHSNoteDraft.from_dict(workflow.get("note_draft") or {})
        image_prompt = str(workflow.get("image_prompt", "")).strip()
        article_title = str(workflow.get("article_title", "")).strip() or note.title
        visibility = str(workflow.get("visibility", "公开可见")).strip() or "公开可见"

        if not image_prompt:
            return {"success": False, "message": "封面图 prompt 还未准备好，暂时不能发布。"}

        negative_prompt = build_cover_negative_prompt(note)
        try:
            image_paths = generate_cover_images(
                note,
                image_count=1,
                cover_core_insight=str(workflow.get("user_cover_prompt", "")).strip(),
                prompt_override=image_prompt,
            )
        except Exception as exc:
            return {"success": False, "message": f"封面图生成失败：{exc}"}

        use_style_transfer = bool(workflow.get("use_style_transfer"))
        if use_style_transfer and image_paths:
            selected_style_path = str(workflow.get("selected_style_path", "")).strip()
            transfer_result = style_transfer_sync(
                content_path=image_paths[0],
                style_path=selected_style_path,
                llm=self.llm,
            )
            if not transfer_result.get("success"):
                return {
                    "success": False,
                    "message": f"风格迁移失败：{transfer_result.get('message', '未知错误')}",
                }
            output_path = str(transfer_result.get("output_path", "")).strip()
            if output_path:
                image_paths = [output_path]
            workflow["style_transfer_result"] = transfer_result

        prepared_payload = self.build_prepared_payload(
            note=note,
            article_title=article_title,
            visibility=visibility,
            image_paths=image_paths,
            is_original=bool(workflow.get("is_original", True)),
        )
        artifact = XHSNoteArtifact(
            note=note,
            image_prompt=image_prompt,
            image_negative_prompt=negative_prompt,
            cover_core_insight=str(workflow.get("user_cover_prompt", "")).strip(),
            image_paths=image_paths,
            image_error="",
            prepared_payload=prepared_payload,
        )
        result = self.publish_generated_note(artifact.to_dict())
        result["artifact"] = artifact.to_dict()
        return result

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
        _log_progress("Generating detailed cover prompt from title")
        image_prompt = self._build_title_based_cover_prompt(note.title)
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
                    cover_core_insight=note.title,
                    supporting_elements=list(note.image_plan.props),
                    prompt_override=image_prompt,
                )
                _log_progress(f"图片生成完成，数量={len(image_paths)}")
            except Exception as exc:
                image_error = str(exc)
                _log_progress(f"图片生成失败：{image_error}")

        prepared_payload = XHSPreparedUploadPayload(
            title=note.title.strip(),
            content=_build_upload_content(note),
            images=image_paths,
            tags=list(dict.fromkeys(note.hashtags)),
            is_original=is_original,
            visibility=visibility,
        )
        artifact = XHSNoteArtifact(
            note=note,
            image_prompt=image_prompt,
            image_negative_prompt=negative_prompt,
            cover_core_insight=note.title,
            image_paths=image_paths,
            image_error=image_error,
            prepared_payload=prepared_payload,
        )
        _log_progress(f"artifact 构建完成，总耗时={time.perf_counter() - overall_start:.2f}s")
        return artifact.to_dict()

    def publish_generated_note(self, artifact: dict) -> dict:
        _log_progress("开始整理上传用标题、正文和封面图")
        prepared_payload = artifact.get("prepared_payload") or {}
        if prepared_payload:
            payload = XHSPreparedUploadPayload(**prepared_payload)
        else:
            note = XHSNoteDraft.from_dict(artifact.get("note") or {})
            payload = XHSPreparedUploadPayload(
                title=note.title.strip(),
                content=_build_upload_content(note),
                images=list(artifact.get("image_paths", []) or []),
                tags=list(dict.fromkeys(note.hashtags)),
                is_original=True,
                visibility="公开可见",
            )

        _log_progress("开始调用小红书 MCP 发布")
        publish_args = build_mcp_publish_args_from_payload(payload)
        result = publish_note_via_mcp_sync(publish_args)
        result["prepared_payload"] = payload.to_dict()
        return result
