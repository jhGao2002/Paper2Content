# session/manager.py
# 负责 session 的生命周期管理：创建、加载、列举、持久化元数据。
# session 元数据（id、标题、创建时间）存储在 sessions.json 中，
# 消息历史通过 SqliteSaver 持久化到 sessions.db，两者通过 session_id 关联。
# 每个 session 完全独立，不同 session 之间不共享任何消息或记忆。

import json
import os
import sqlite3
from datetime import datetime

SESSIONS_FILE = os.path.join(os.path.dirname(__file__), "..", "sessions.json")
SESSIONS_DB = os.path.join(os.path.dirname(__file__), "..", "sessions.db")


def _normalize_session(session_id: str, info: dict) -> dict:
    return {
        "user_id": info.get("user_id", ""),
        "title": info.get("title") or "新对话",
        "created_at": info.get("created_at") or datetime.now().isoformat(),
        "selected_documents": list(info.get("selected_documents", [])),
        "selected_style_image": str(info.get("selected_style_image", "")).strip(),
        "publish_workflow": info.get("publish_workflow") if isinstance(info.get("publish_workflow"), dict) else None,
    }


def _load_sessions() -> dict:
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {
            session_id: _normalize_session(session_id, info)
            for session_id, info in raw.items()
        }
    return {}


def _save_sessions(sessions: dict):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def create_session(user_id: str, title: str = "") -> str:
    """创建新 session，返回 session_id。"""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sessions = _load_sessions()
    sessions[session_id] = {
        "user_id": user_id,
        "title": title or "新对话",
        "created_at": datetime.now().isoformat(),
        "selected_documents": [],
        "selected_style_image": "",
        "publish_workflow": None,
    }
    _save_sessions(sessions)
    print(f"[Session] 新建会话: {session_id}")
    return session_id


def update_session_title(session_id: str, first_message: str):
    """用第一条用户消息的前 20 字作为会话标题。"""
    sessions = _load_sessions()
    if session_id in sessions and sessions[session_id]["title"] == "新对话":
        sessions[session_id]["title"] = first_message[:20]
        _save_sessions(sessions)


def list_sessions(user_id: str) -> list:
    """返回该用户的所有 session，按创建时间倒序。"""
    sessions = _load_sessions()
    user_sessions = [
        {"session_id": sid, **info}
        for sid, info in sessions.items()
        if info.get("user_id") == user_id
    ]
    return sorted(user_sessions, key=lambda x: x["created_at"], reverse=True)


def get_session(session_id: str) -> dict | None:
    sessions = _load_sessions()
    return sessions.get(session_id)


def get_session_documents(session_id: str) -> list[str]:
    session = get_session(session_id)
    if not session:
        return []
    return list(session.get("selected_documents", []))


def get_session_style_image(session_id: str) -> str:
    session = get_session(session_id)
    if not session:
        return ""
    return str(session.get("selected_style_image", "")).strip()


def get_publish_workflow(session_id: str) -> dict | None:
    session = get_session(session_id)
    if not session:
        return None
    workflow = session.get("publish_workflow")
    if isinstance(workflow, dict):
        return workflow
    return None


def set_publish_workflow(session_id: str, workflow: dict | None) -> bool:
    sessions = _load_sessions()
    if session_id not in sessions:
        return False
    sessions[session_id]["publish_workflow"] = workflow if isinstance(workflow, dict) else None
    _save_sessions(sessions)
    return True


def clear_publish_workflow(session_id: str) -> bool:
    return set_publish_workflow(session_id, None)


def set_session_documents(session_id: str, filenames: list[str]) -> bool:
    sessions = _load_sessions()
    if session_id not in sessions:
        return False
    sessions[session_id]["selected_documents"] = list(dict.fromkeys(filenames))
    _save_sessions(sessions)
    return True


def set_session_style_image(session_id: str, filename: str) -> bool:
    sessions = _load_sessions()
    if session_id not in sessions:
        return False
    sessions[session_id]["selected_style_image"] = str(filename or "").strip()
    _save_sessions(sessions)
    return True


def remove_document_from_all_sessions(filename: str):
    sessions = _load_sessions()
    changed = False
    for session in sessions.values():
        selected = session.get("selected_documents", [])
        if filename not in selected:
            continue
        session["selected_documents"] = [name for name in selected if name != filename]
        changed = True
    if changed:
        _save_sessions(sessions)


def remove_style_image_from_all_sessions(filename: str):
    sessions = _load_sessions()
    changed = False
    for session in sessions.values():
        if str(session.get("selected_style_image", "")).strip() != filename:
            continue
        session["selected_style_image"] = ""
        changed = True
    if changed:
        _save_sessions(sessions)


def delete_session(session_id: str) -> bool:
    sessions = _load_sessions()
    removed = sessions.pop(session_id, None)
    if removed is None:
        return False

    _save_sessions(sessions)

    if os.path.exists(SESSIONS_DB):
        conn = sqlite3.connect(SESSIONS_DB)
        try:
            conn.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()

    print(f"[Session] 已删除会话: {session_id}")
    return True


def get_db_path() -> str:
    return os.path.abspath(SESSIONS_DB)
