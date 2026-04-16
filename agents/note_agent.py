# agents/note_agent.py
# 定义 NoteAgent 的工具和系统提示。
# recall_notes：从向量记忆中检索本会话历史笔记和自动提取的事实。
# save_note：将用户明确要求记录的内容保存为笔记。
# publish_graphic_note：将当前会话中的文献 insight 整理为小红书图文笔记、生成配图并自动发布。

from agents.base import build_sub_agent
from langchain_core.tools import tool

from memory.store import save_fact


SYSTEM_PROMPT = (
    "你是用户的笔记整理与发布助手。\n"
    "使用 recall_notes 查询本会话的历史笔记和重要事实，"
    "使用 save_note 保存用户明确要求记录的新笔记。\n"
    "如果用户要求整理图文笔记、发布图文笔记、发布到小红书、自动生成配图后发布，"
    "必须调用 publish_graphic_note 执行自动化发布。\n"
    "如果当前对话里已经有上一步整理出的文献 insight，请把这些 insight 提炼后放进 publish_graphic_note 的 extra_context 参数。\n"
    "每次回答只需调用一次工具，调用后直接基于结果作答。"
)


def _build_note_history(note_history_provider, extra_context: str) -> list[dict]:
    history = list(note_history_provider() or []) if note_history_provider else []
    extra = extra_context.strip()
    if extra:
        history.append({"role": "assistant", "content": extra})
    return history


def make_note_agent(
    llm,
    memory_store,
    user_id: str,
    session_id: str,
    stats: dict,
    note_service,
    note_history_provider,
):
    @tool
    def recall_notes(query: str, note_type: str = "all") -> str:
        """从本会话的历史笔记和自动提取事实中检索相关内容。
        note_type 可选：'note'（手动笔记）| 'auto_fact'（自动提取）| 'all'（全部）
        """
        print(f"  [NoteAgent][TOOL] recall_notes 开始，query: {query}, type: {note_type}")
        must_filters = [
            {"key": "user_id", "match": {"value": user_id}},
            {"key": "session_id", "match": {"value": session_id}},
        ]
        if note_type != "all":
            must_filters.append({"key": "type", "match": {"value": note_type}})
        try:
            docs = memory_store.similarity_search(query, k=5, filter={"must": must_filters})
        except Exception as exc:
            print(f"  [NoteAgent][TOOL] 检索失败: {exc}")
            docs = []

        if not docs:
            print("  [NoteAgent][TOOL] 未找到相关笔记")
            return "未找到相关笔记。"

        results = []
        for doc in docs:
            ts = doc.metadata.get("timestamp", "")
            doc_type = doc.metadata.get("type", "note")
            results.append(f"[{doc_type}{' | ' + ts[:10] if ts else ''}]: {doc.page_content}")
        print(f"  [NoteAgent][TOOL] 找到 {len(results)} 条笔记")
        return "\n\n".join(results)

    @tool
    def save_note(content: str) -> str:
        """将用户提供的内容保存为学习笔记到本会话记忆库。"""
        print(f"  [NoteAgent][TOOL] save_note: {content[:50]}")
        save_fact(memory_store, content, user_id, session_id, fact_type="note")
        stats["notes_added"] += 1
        return f"笔记已保存：{content[:50]}..."

    @tool
    def publish_graphic_note(extra_context: str = "", visibility: str = "公开可见") -> str:
        """将当前会话中的文献 insight 整理成小红书图文笔记、生成配图并自动发布。"""
        if note_service is None:
            return "当前环境未初始化图文笔记发布服务，暂时无法自动发布。"

        history = _build_note_history(note_history_provider, extra_context)
        if not history:
            return "当前会话里还没有可整理的问答记录，暂时无法发布图文笔记。"

        print("  [NoteAgent][TOOL] publish_graphic_note 开始执行")
        artifact = note_service.generate_note_artifact(
            history=history,
            generate_images=True,
            image_count=1,
            visibility=visibility,
        )
        note = artifact["note"]

        if artifact["image_error"] or not artifact["image_paths"]:
            return (
                f"图文笔记已整理完成，但自动发布中止。\n"
                f"标题：{note['title']}\n"
                f"摘要：{note['summary']}\n"
                f"正文：{note['body']}\n"
                f"配图生成失败：{artifact['image_error'] or '未生成图片'}\n"
                f"封面提示词：{artifact['image_prompt']}"
            )

        try:
            result = note_service.publish_generated_note(artifact)
        except Exception as exc:
            return (
                f"图文笔记和配图已生成，但发布失败：{exc}\n"
                f"标题：{note['title']}\n"
                f"正文：{note['body']}\n"
                f"图片：{', '.join(artifact['image_paths'])}"
            )

        if result.get("success"):
            return (
                f"小红书图文笔记已自动发布成功。\n"
                f"标题：{note['title']}\n"
                f"摘要：{note['summary']}\n"
                f"正文：{note['body']}\n"
                f"图片：{', '.join(artifact['image_paths'])}\n"
                f"返回信息：{result.get('message', '发布成功')}"
            )

        return (
            f"图文笔记和配图已生成，但发布失败。\n"
            f"标题：{note['title']}\n"
            f"正文：{note['body']}\n"
            f"图片：{', '.join(artifact['image_paths'])}\n"
            f"返回信息：{result.get('message', '未知错误')}"
        )

    return build_sub_agent(
        llm,
        [recall_notes, save_note, publish_graphic_note],
        SYSTEM_PROMPT,
        name="NoteAgent",
        max_tool_calls=1,
    )
