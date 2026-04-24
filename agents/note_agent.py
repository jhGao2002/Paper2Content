from __future__ import annotations

import json
import re
from typing import Callable, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from agents.agent_utils import (
    PublishWorkflowContext,
    extract_explicit_title,
    is_cancel_message,
    is_confirm_message,
    is_continue_message,
    is_no_message,
    is_publish_intent,
    is_yes_message,
    last_user_message,
    messages_to_history,
)
from agents.base import build_sub_agent
from memory.store import save_fact
from session.manager import (
    clear_publish_workflow,
    get_publish_workflow,
    get_session_style_image,
    set_publish_workflow,
)
from style.gallery import get_style_image_path


SYSTEM_PROMPT = (
    "你是用户的笔记整理与发布助手。\n"
    "使用 recall_notes 查询本会话的历史笔记和重要事实，"
    "使用 save_note 保存用户明确要求记录的新笔记。\n"
    "如果用户是要查询历史笔记、保存笔记，就正常使用工具回答。"
)


def _log_progress(message: str) -> None:
    print(f"  [NoteAgent] {message}", flush=True)


class PublishGraphState(TypedDict):
    context: PublishWorkflowContext
    response: str


class NoteWorkflowAgent:
    def __init__(
        self,
        llm,
        memory_store,
        user_id: str,
        session_id: str,
        stats: dict,
        note_service,
    ):
        self.llm = llm
        self.memory_store = memory_store
        self.user_id = user_id
        self.session_id = session_id
        self.stats = stats
        self.note_service = note_service
        self.fallback_agent = self._build_fallback_agent(llm)
        self.workflow_graph = self._build_workflow_graph()

    def _build_fallback_agent(self, llm):
        @tool
        def recall_notes(query: str, note_type: str = "all") -> str:
            """从本会话的历史笔记和自动提取事实中检索相关内容。"""
            must_filters = [
                {"key": "user_id", "match": {"value": self.user_id}},
                {"key": "session_id", "match": {"value": self.session_id}},
            ]
            if note_type != "all":
                must_filters.append({"key": "type", "match": {"value": note_type}})
            try:
                docs = self.memory_store.similarity_search(query, k=5, filter={"must": must_filters})
            except Exception as exc:
                return f"检索笔记失败：{exc}"

            if not docs:
                return "未找到相关笔记。"

            results = []
            for doc in docs:
                ts = doc.metadata.get("timestamp", "")
                doc_type = doc.metadata.get("type", "note")
                results.append(f"[{doc_type}{' | ' + ts[:10] if ts else ''}]: {doc.page_content}")
            return "\n\n".join(results)

        @tool
        def save_note(content: str) -> str:
            """将用户提供的内容保存为学习笔记到本会话记忆库。"""
            save_fact(self.memory_store, content, self.user_id, self.session_id, fact_type="note")
            self.stats["notes_added"] += 1
            return f"笔记已保存：{content[:50]}..."

        return build_sub_agent(
            llm,
            [recall_notes, save_note],
            SYSTEM_PROMPT,
            name="NoteAgent",
            max_tool_calls=1,
        )

    def _workflow_handlers(self) -> dict[str, Callable[[PublishWorkflowContext], str]]:
        return {
            "await_selection": self._run_selection_node,
            "await_body_confirmation": self._run_body_confirmation_node,
            "await_cover_prompt": self._run_cover_prompt_node,
            "await_style_transfer_decision": self._run_style_transfer_decision_node,
            "await_confirmation": self._run_confirmation_node,
        }

    def _build_workflow_graph(self):
        # LangGraph 负责调度当前发布阶段；每轮用户输入只推进一个节点。
        graph = StateGraph(PublishGraphState)
        handlers = self._workflow_handlers()

        graph.add_node("dispatch", lambda state: state)
        graph.add_node("invalid_stage", self._run_invalid_stage_node)
        for stage, handler in handlers.items():
            graph.add_node(stage, self._graph_node(handler))

        graph.add_edge(START, "dispatch")
        graph.add_conditional_edges(
            "dispatch",
            self._route_workflow_stage,
            {**{stage: stage for stage in handlers}, "invalid_stage": "invalid_stage"},
        )
        for stage in handlers:
            graph.add_edge(stage, END)
        graph.add_edge("invalid_stage", END)
        return graph.compile()

    def _route_workflow_stage(self, state: PublishGraphState) -> str:
        stage = str(state["context"].workflow.get("stage", ""))
        return stage if stage in self._workflow_handlers() else "invalid_stage"

    def _graph_node(self, handler: Callable[[PublishWorkflowContext], str]):
        def run(state: PublishGraphState) -> dict:
            return {"response": handler(state["context"])}

        return run

    def _run_invalid_stage_node(self, state: PublishGraphState) -> dict:
        clear_publish_workflow(self.session_id)
        return {
            "response": "发布流程状态异常，我先帮你重置了。你可以重新说“发布到小红书”开始。"
        }

    def _memory_note_candidates(self) -> list[str]:
        must_filters = [
            {"key": "user_id", "match": {"value": self.user_id}},
            {"key": "session_id", "match": {"value": self.session_id}},
            {"key": "type", "match": {"value": "note"}},
        ]
        try:
            docs = self.memory_store.get_documents(filter={"must": must_filters}, limit=20)
        except Exception:
            return []
        return [str(doc.page_content).strip() for doc in docs if str(doc.page_content).strip()]

    def _start_publish_workflow(self, history: list[dict]) -> str:
        candidates = self.note_service.build_publish_candidates(
            history=history,
            memory_notes=self._memory_note_candidates(),
        )
        if not candidates:
            clear_publish_workflow(self.session_id)
            return self.note_service.render_candidate_list(candidates)

        workflow = {
            "active": True,
            "stage": "await_selection",
            "candidates": candidates,
            "selected_items": [],
            "selected_indexes": [],
            "article_title": "",
            "note_draft": None,
            "user_cover_prompt": "",
            "image_prompt": "",
            "visibility": "公开可见",
            "is_original": True,
            "use_style_transfer": False,
            "selected_style_image": "",
            "selected_style_path": "",
        }
        set_publish_workflow(self.session_id, workflow)
        return self.note_service.render_candidate_list(candidates)

    def invoke(self, payload: dict) -> dict:
        messages = list(payload.get("messages", []))
        user_message = last_user_message(messages)
        workflow = get_publish_workflow(self.session_id) or {}

        if is_publish_intent(user_message) or workflow.get("active"):
            response = self._handle_publish_flow(messages, workflow, user_message)
            return {"messages": [AIMessage(content=response)]}

        return self.fallback_agent.invoke(payload)

    def _handle_publish_flow(self, messages: list, workflow: dict, user_message: str) -> str:
        if self.note_service is None:
            return "当前环境未初始化图文笔记服务，暂时无法处理小红书发布流程。"

        history = messages_to_history(messages)

        if is_cancel_message(user_message):
            clear_publish_workflow(self.session_id)
            return "已取消当前小红书发布流程。后续如果你想重新发布，直接再说“发布到小红书”即可。"

        if is_publish_intent(user_message) and not workflow.get("active"):
            _log_progress("进入发布流程，开始列出可用素材")
            return self._start_publish_workflow(history)

        if not workflow.get("active"):
            return "如果你想发布小红书图文笔记，可以直接说“发布到小红书”。"

        stage = str(workflow.get("stage", ""))
        _log_progress(f"处理发布流程阶段：{stage}")

        note = None
        article_title = ""
        if stage in self._workflow_handlers() and stage != "await_selection":
            note_data = workflow.get("note_draft")
            if not isinstance(note_data, dict):
                clear_publish_workflow(self.session_id)
                return "发布草稿状态已失效，请重新说一次“发布到小红书”。"

            from xhs.schemas import XHSNoteDraft

            note = XHSNoteDraft.from_dict(note_data)
            article_title = str(workflow.get("article_title", "")).strip() or note.title

        context = PublishWorkflowContext(
            messages=messages,
            history=history,
            workflow=workflow,
            user_message=user_message,
            note=note,
            article_title=article_title,
        )
        result = self.workflow_graph.invoke({"context": context, "response": ""})
        return str(result.get("response", ""))

    def _run_selection_node(self, context: PublishWorkflowContext) -> str:
        indexes = list(dict.fromkeys(int(item) for item in re.findall(r"\d+", context.user_message)))
        candidates = context.workflow.get("candidates", []) or []
        selected_items = [
            candidates[index - 1]
            for index in indexes
            if 1 <= index <= len(candidates)
        ]
        if not selected_items:
            return "我还没收到有效编号。请直接回复素材编号，例如：`1,3`。"

        note, article_title = self.note_service.build_publish_note_from_candidates(selected_items)
        context.workflow.update(
            {
                "stage": "await_body_confirmation",
                "selected_items": selected_items,
                "selected_indexes": indexes,
                "note_draft": note.to_dict(),
                "article_title": article_title,
            }
        )
        set_publish_workflow(self.session_id, context.workflow)
        return (
            "我已经根据你选中的素材整理出一版小红书正文草稿。\n\n"
            f"{note.body}\n\n"
            "如果你还想改正文，直接告诉我修改意见就行。\n"
            "如果这版正文可以了，想继续做封面，就回复“继续做封面”或“进入下一步”。"
        )

    def _run_body_confirmation_node(self, context: PublishWorkflowContext) -> str:
        workflow = context.workflow
        note = context.note
        user_message = context.user_message

        if is_continue_message(user_message):
            workflow["stage"] = "await_cover_prompt"
            set_publish_workflow(self.session_id, workflow)
            return (
                "好的，我们进入封面图阶段。\n"
                "请给我一个封面图的初步 prompt。\n"
                "这个 prompt 可以是你觉得这次会话里最重要的知识点、核心 insight，"
                "或者你最想让封面突出的那一句话。"
            )

        explicit_title = extract_explicit_title(user_message)
        if explicit_title:
            note.title = explicit_title[:20]

        extracted_tags = re.findall(r"#([^\s#]+)", user_message)
        if extracted_tags:
            note.hashtags = list(dict.fromkeys(tag.strip() for tag in extracted_tags if tag.strip()))[:6]

        note = self.note_service.revise_publish_note(note, context.article_title, user_message)
        workflow["note_draft"] = note.to_dict()
        set_publish_workflow(self.session_id, workflow)
        return (
            "我已经按你的意见更新了图文笔记草稿。\n\n"
            f"{note.body}\n\n"
            "如果还想改正文，继续告诉我修改意见就行；"
            "如果正文可以了，想继续做封面，就回复“继续做封面”或“进入下一步”。"
        )

    def _run_cover_prompt_node(self, context: PublishWorkflowContext) -> str:
        workflow = context.workflow
        note = context.note
        advanced_prompt = self.note_service.build_cover_prompt_from_user_prompt(note, context.user_message)
        selected_style_image = get_session_style_image(self.session_id)
        selected_style_path = get_style_image_path(selected_style_image) or ""
        workflow.update(
            {
                "stage": "await_style_transfer_decision",
                "note_draft": note.to_dict(),
                "user_cover_prompt": context.user_message.strip(),
                "image_prompt": advanced_prompt,
                "selected_style_image": selected_style_image,
                "selected_style_path": selected_style_path,
            }
        )
        set_publish_workflow(self.session_id, workflow)
        style_hint = self._style_hint(selected_style_image)
        return (
            "封面图高级 prompt 已准备好。\n"
            f"{style_hint}\n"
            "接下来是否要对刚生成的封面图进行风格迁移？\n"
            "你可以回复“需要风格迁移”或“不需要风格迁移”。"
        )

    def _style_hint(self, selected_style_image: str) -> str:
        if selected_style_image == "__remote_default__":
            return "当前会使用远程 MCP 服务的默认风格图。"
        if selected_style_image:
            return f"当前会使用你选择的风格图：{selected_style_image}"
        return "当前没有选择本地风格图；如果需要风格迁移，将由 MCP 服务使用默认风格图。"

    def _run_style_transfer_decision_node(self, context: PublishWorkflowContext) -> str:
        decision = self._classify_style_transfer_intent(context.user_message)
        if decision is None:
            return "我还没判断出你的选择。请明确回复“需要风格迁移”或“不需要风格迁移”。"

        workflow = context.workflow
        workflow["use_style_transfer"] = decision
        workflow["stage"] = "await_confirmation"
        workflow["selected_style_image"] = get_session_style_image(self.session_id)
        workflow["selected_style_path"] = get_style_image_path(workflow["selected_style_image"]) or ""
        set_publish_workflow(self.session_id, workflow)
        return self.note_service.render_publish_confirmation(workflow)

    def _classify_style_transfer_intent(self, user_message: str) -> bool | None:
        if is_yes_message(user_message) and not is_no_message(user_message):
            return True
        if is_no_message(user_message):
            return False

        prompt = (
            "判断用户是否希望对刚生成的小红书封面图进行风格迁移。"
            "如果用户表达想要换风格、统一风格、使用风格图、把封面图做风格化，返回 true；"
            "如果用户拒绝或跳过，返回 false；无法判断返回 null。"
            "只输出 JSON：{\"use_style_transfer\": true/false/null}。\n\n"
            f"用户回复：{user_message}"
        )
        try:
            raw = str(self.llm.invoke(prompt).content).strip()
            start = raw.find("{")
            end = raw.rfind("}")
            payload = json.loads(raw[start : end + 1]) if start >= 0 and end > start else {}
            value = payload.get("use_style_transfer")
            return value if isinstance(value, bool) else None
        except Exception:
            return None

    def _run_confirmation_node(self, context: PublishWorkflowContext) -> str:
        workflow = context.workflow
        user_message = context.user_message
        note = context.note

        if is_confirm_message(user_message):
            result = self.note_service.publish_confirmed_workflow(workflow)
            if result.get("success"):
                clear_publish_workflow(self.session_id)
                prepared = result.get("prepared_payload") or {}
                return (
                    "小红书图文笔记已发布成功。\n"
                    f"标题：{prepared.get('title', note.title)}\n"
                    f"正文：{prepared.get('content', note.body)}\n"
                    f"图片：{', '.join((result.get('artifact') or {}).get('image_paths', [])) or '未返回'}\n"
                    f"返回信息：{result.get('message', '发布成功')}"
                )
            set_publish_workflow(self.session_id, workflow)
            return f"发布失败：{result.get('message', '未知错误')}\n你可以修改草稿后再次确认发布。"

        updated, note = self._apply_confirmation_edits(workflow, note, context.article_title, user_message)
        note.body = self.note_service._finalize_publish_body(note.body, context.article_title)
        workflow["note_draft"] = note.to_dict()
        set_publish_workflow(self.session_id, workflow)

        if updated:
            return "我已经按你的意见更新发布草稿。\n\n" + self.note_service.render_publish_confirmation(workflow)
        return (
            "我这边还没有收到“确认发布”的明确信号。\n"
            "如果你想继续调整，直接告诉我修改意见；如果没问题，再回复“确认发布”。"
        )

    def _apply_confirmation_edits(self, workflow: dict, note, article_title: str, user_message: str) -> tuple[bool, object]:
        updated = False
        visibility = workflow.get("visibility", "公开可见")
        new_visibility = self.note_service.build_prepared_payload(
            note=note,
            article_title=article_title,
            visibility=user_message,
        ).visibility
        if new_visibility != visibility:
            workflow["visibility"] = new_visibility
            updated = True

        explicit_title = extract_explicit_title(user_message)
        if explicit_title:
            note.title = explicit_title[:20]
            updated = True

        extracted_tags = re.findall(r"#([^\s#]+)", user_message)
        if extracted_tags:
            note.hashtags = list(dict.fromkeys(tag.strip() for tag in extracted_tags if tag.strip()))[:6]
            updated = True

        if any(keyword in user_message for keyword in ("正文", "改短", "改长", "压缩", "扩写", "重写", "语气", "标签", "标题")):
            note = self.note_service.revise_publish_note(note, article_title, user_message)
            workflow["note_draft"] = note.to_dict()
            updated = True

        if any(keyword in user_message.lower() for keyword in ("封面", "prompt")):
            workflow["image_prompt"] = self.note_service.revise_cover_prompt(
                note,
                str(workflow.get("image_prompt", "")).strip(),
                user_message,
            )
            if "prompt" in user_message.lower() or "封面" in user_message:
                workflow["user_cover_prompt"] = user_message.strip()
            updated = True

        if "风格迁移" in user_message or "风格化" in user_message:
            decision = self._classify_style_transfer_intent(user_message)
            if decision is not None:
                workflow["use_style_transfer"] = decision
                updated = True
            workflow["selected_style_image"] = get_session_style_image(self.session_id)
            workflow["selected_style_path"] = get_style_image_path(workflow["selected_style_image"]) or ""

        return updated, note


def make_note_agent(
    llm,
    memory_store,
    user_id: str,
    session_id: str,
    stats: dict,
    note_service,
    note_history_provider,
):
    _ = note_history_provider
    return NoteWorkflowAgent(
        llm=llm,
        memory_store=memory_store,
        user_id=user_id,
        session_id=session_id,
        stats=stats,
        note_service=note_service,
    )
