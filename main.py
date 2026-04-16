import os
from datetime import datetime
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from config import get_embeddings, get_fast_llm, get_llm
from document.loader import load_document
from graph.builder import build_graph
from memory.compression import compress_window
from memory.store import build_memory_context, init_vector_stores
from session.manager import create_session, get_db_path, update_session_title
from xhs.note_service import XHSNoteService


def _env_flag(name: str, default: bool = False) -> bool:
    value = str(default) if os.getenv(name) is None else os.getenv(name, str(default))
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        self.loaded_docs: list[str] = []
        self._first_message = True
        self.last_retrieved_docs = ""
        self.xhs_note_service = XHSNoteService(llm=self.llm)

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
        )

    def _record_retrieval(self, content: str) -> None:
        self.last_retrieved_docs = content

    def get_chat_history(self) -> list:
        config = {"configurable": {"thread_id": self.session_id}}
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
        config = {"configurable": {"thread_id": self.session_id}}

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
        return result

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

    demo = create_gradio_ui(app_factory=MultiAgentApp)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=_env_flag("GRADIO_SHARE", False),
        quiet=True,
    )
