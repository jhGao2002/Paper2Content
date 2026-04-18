import os
from datetime import datetime
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from config import build_langsmith_runnable_config, get_embeddings, get_fast_llm, get_llm
from document.loader import load_document
from document.registry import get_all_documents, unregister_document
from document.source_excerpt import collect_cover_source_materials
from graph.builder import build_graph
from memory.compression import compress_window
from memory.store import build_memory_context, init_vector_stores
from session.manager import (
    create_session,
    delete_session,
    get_db_path,
    get_publish_workflow,
    get_session_documents,
    get_session_style_image,
    remove_document_from_all_sessions,
    remove_style_image_from_all_sessions,
    set_session_documents,
    set_session_style_image,
    update_session_title,
)
from style.gallery import delete_style_image, get_style_image_path, list_style_images, save_style_image
from xhs.note_service import XHSNoteService

DEFAULT_REMOTE_STYLE_VALUE = "默认"


def _env_flag(name: str, default: bool = False) -> bool:
    value = str(default) if os.getenv(name) is None else os.getenv(name, str(default))
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_text(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value.strip() or default


def _ensure_local_no_proxy() -> None:
    local_hosts = ["127.0.0.1", "localhost", "0.0.0.0"]
    for env_name in ("NO_PROXY", "no_proxy"):
        current = os.getenv(env_name, "").strip()
        entries = [item.strip() for item in current.split(",") if item.strip()]
        for host in local_hosts:
            if host not in entries:
                entries.append(host)
        os.environ[env_name] = ",".join(entries)


def _launch_demo(demo) -> None:
    share = _env_flag("GRADIO_SHARE", False)
    port = int(_env_text("GRADIO_SERVER_PORT", "7861"))
    configured_host = _env_text("GRADIO_SERVER_NAME", "127.0.0.1")
    fallback_host = "127.0.0.1"
    _ensure_local_no_proxy()

    launch_hosts = [configured_host]
    if configured_host != fallback_host:
        launch_hosts.append(fallback_host)

    last_error: Exception | None = None
    for index, host in enumerate(launch_hosts, start=1):
        print(
            f"[Startup] 准备启动 Gradio（第{index}次），host={host}，port={port}，share={share}",
            flush=True,
        )
        print(
            f"[Startup] NO_PROXY={os.getenv('NO_PROXY', '') or os.getenv('no_proxy', '')}",
            flush=True,
        )
        try:
            demo.launch(
                server_name=host,
                server_port=port,
                share=share,
                quiet=False,
                show_error=True,
            )
            return
        except Exception as exc:
            last_error = exc
            print(f"[Startup] Gradio 启动失败，host={host}，error={exc}", flush=True)

    if last_error is not None:
        raise last_error


class MultiAgentApp:
    def __init__(self, user_id: str = "default_user", session_id: str | None = None):
        self.user_id = user_id
        self.session_id = session_id or create_session(user_id)

        self.llm = get_llm()
        self.fast_llm = get_fast_llm()
        embeddings = get_embeddings()
        self.vector_backend, self.pdf_store, self.memory_store = init_vector_stores(embeddings)

        self.stats = {
            "session_start": datetime.now(),
            "docs_loaded": 0,
            "questions_asked": 0,
            "notes_added": 0,
        }
        valid_names = {doc["filename"] for doc in get_all_documents()}
        self.loaded_docs: list[str] = [
            name for name in get_session_documents(self.session_id)
            if name in valid_names
        ]
        set_session_documents(self.session_id, self.loaded_docs)
        self.selected_style_image = get_session_style_image(self.session_id)
        if self.selected_style_image and self.selected_style_image != DEFAULT_REMOTE_STYLE_VALUE and not get_style_image_path(self.selected_style_image):
            self.selected_style_image = ""
            set_session_style_image(self.session_id, "")
        self._first_message = True
        self.last_retrieved_docs = ""
        self.xhs_note_service = XHSNoteService(
            llm=self.llm,
            fast_llm=self.fast_llm,
            cover_source_provider=self._get_cover_source_materials,
        )

        conn = sqlite3.connect(get_db_path(), check_same_thread=False)
        self._db_conn = SqliteSaver(conn)
        self.app = build_graph(
            llm=self.llm,
            fast_llm=self.fast_llm,
            pdf_store=self.pdf_store,
            memory_store=self.memory_store,
            user_id=self.user_id,
            session_id=self.session_id,
            stats=self.stats,
            loaded_docs=self.loaded_docs,
            checkpointer=self._db_conn,
            on_retrieval=self._record_retrieval,
            note_service=self.xhs_note_service,
            note_history_provider=self.get_chat_history,
            has_active_publish_workflow=self.has_active_publish_workflow,
        )

    def _runtime_config(
        self,
        *,
        run_name: str,
        extra_tags: list[str] | None = None,
        extra_metadata: dict | None = None,
    ) -> dict:
        metadata = {
            "component": "multi_agent_app",
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return build_langsmith_runnable_config(
            run_name=run_name,
            extra_tags=["runtime", "session", *(extra_tags or [])],
            extra_metadata=metadata,
            configurable={"thread_id": self.session_id},
        )

    def _record_retrieval(self, content: str) -> None:
        self.last_retrieved_docs = content

    def _get_cover_source_materials(self) -> list[dict]:
        return collect_cover_source_materials(
            self.pdf_store,
            self.loaded_docs,
            user_id=self.user_id,
        )

    def get_chat_history(self) -> list:
        config = self._runtime_config(run_name="chat_history_read", extra_tags=["history"])
        try:
            state = self.app.get_state(config)
            messages = state.values.get("messages", [])
        except Exception:
            return []

        history = []
        pending_user = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                pending_user = msg.content
            elif isinstance(msg, AIMessage) and msg.content and getattr(msg, "name", None):
                if pending_user is not None:
                    history.append({"role": "user", "content": pending_user})
                    history.append({"role": "assistant", "content": msg.content})
                    pending_user = None
        return history

    def ask(self, question: str) -> str:
        self.stats["questions_asked"] += 1
        self.last_retrieved_docs = ""
        config = self._runtime_config(
            run_name="chat_turn",
            extra_tags=["chat"],
            extra_metadata={"question_length": len(question)},
        )

        if self._first_message:
            update_session_title(self.session_id, question)
            self._first_message = False

        compress_window(
            self.app,
            config,
            self.memory_store,
            self.user_id,
            self.session_id,
            self.fast_llm,
        )

        input_messages = []
        context = build_memory_context(self.memory_store, self.user_id, self.session_id, question)
        if context:
            input_messages.append(SystemMessage(content=context))
        input_messages.append(HumanMessage(content=question))

        result = self.app.invoke(
            {
                "messages": input_messages,
                "plan": [],
                "plan_step": 0,
                "step_results": [],
            },
            config=config,
        )

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return "未生成有效回答。"

    def load_document(self, pdf_path: str) -> dict:
        result = load_document(pdf_path, self.pdf_store, self.user_id, fast_llm=self.fast_llm)
        if result["success"]:
            self.stats["docs_loaded"] += 1
            doc_name = result["document"]
            if doc_name not in self.loaded_docs:
                self.loaded_docs.append(doc_name)
                set_session_documents(self.session_id, self.loaded_docs)
        return result

    def list_available_documents(self) -> list[dict]:
        return get_all_documents()

    def set_session_documents(self, filenames: list[str]) -> list[str]:
        valid_names = {doc["filename"] for doc in get_all_documents()}
        self.loaded_docs[:] = [name for name in dict.fromkeys(filenames) if name in valid_names]
        set_session_documents(self.session_id, self.loaded_docs)
        self.stats["docs_loaded"] = len(self.loaded_docs)
        return list(self.loaded_docs)

    def delete_document(self, filename: str) -> dict:
        if not filename:
            return {"success": False, "message": "请先选择要删除的文档。"}

        deleted_chunks = self.pdf_store.delete_documents(
            filter={"must": [{"key": "source", "match": {"value": filename}}]}
        )
        removed_meta = unregister_document(filename)
        remove_document_from_all_sessions(filename)
        self.loaded_docs[:] = [name for name in self.loaded_docs if name != filename]
        self.stats["docs_loaded"] = len(self.loaded_docs)

        if not deleted_chunks and not removed_meta:
            return {"success": False, "message": f"未找到文档：{filename}"}

        return {
            "success": True,
            "message": f"已删除文档 {filename}，清理 {deleted_chunks} 个向量分块。",
        }

    def delete_current_session(self) -> dict:
        deleted_memory = self.memory_store.delete_documents(
            filter={
                "must": [
                    {"key": "user_id", "match": {"value": self.user_id}},
                    {"key": "session_id", "match": {"value": self.session_id}},
                ]
            }
        )
        removed = delete_session(self.session_id)
        if not removed:
            return {"success": False, "message": f"未找到会话：{self.session_id}"}

        return {
            "success": True,
            "message": f"已删除会话 {self.session_id}，清理 {deleted_memory} 条长期记忆。",
        }

    def has_active_publish_workflow(self) -> bool:
        workflow = get_publish_workflow(self.session_id)
        return bool(isinstance(workflow, dict) and workflow.get("active"))

    def list_style_images(self) -> list[dict]:
        return list_style_images()

    def set_session_style_image(self, filename: str) -> str:
        valid_names = {item["filename"] for item in list_style_images()}
        selected = str(filename or "").strip()
        if selected == DEFAULT_REMOTE_STYLE_VALUE:
            pass
        elif selected and selected not in valid_names:
            selected = ""
        self.selected_style_image = selected
        set_session_style_image(self.session_id, selected)
        return self.selected_style_image

    def upload_style_image(self, image_path: str) -> dict:
        result = save_style_image(image_path)
        if result.get("success"):
            self.set_session_style_image(str(result.get("filename", "")))
        return result

    def delete_style_image(self, filename: str) -> dict:
        if not filename:
            return {"success": False, "message": "请先选择要删除的风格图。"}
        deleted = delete_style_image(filename)
        if not deleted:
            return {"success": False, "message": f"未找到风格图：{filename}"}
        remove_style_image_from_all_sessions(filename)
        if self.selected_style_image == filename:
            self.selected_style_image = ""
        return {"success": True, "message": f"已删除风格图：{filename}"}

    def get_selected_style_image_path(self) -> str:
        if self.selected_style_image == DEFAULT_REMOTE_STYLE_VALUE:
            return ""
        return get_style_image_path(self.selected_style_image) or ""

    def generate_xhs_note(
        self,
        generate_images: bool = False,
        image_count: int = 1,
        output_dir: str | None = None,
        is_original: bool = True,
        visibility: str = "公开可见",
    ) -> dict:
        history = self.get_chat_history()
        return self.xhs_note_service.generate_note_artifact(
            history=history,
            generate_images=generate_images,
            image_count=image_count,
            output_dir=output_dir,
            is_original=is_original,
            visibility=visibility,
        )

    def publish_xhs_note(self, artifact: dict) -> dict:
        return self.xhs_note_service.publish_generated_note(artifact)


if __name__ == "__main__":
    from ui.gradio_app import create_gradio_ui

    print("[Startup] 正在构建 Gradio UI...", flush=True)
    demo = create_gradio_ui(app_factory=MultiAgentApp)
    print("[Startup] Gradio UI 构建完成。", flush=True)
    _launch_demo(demo)
