from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage


PUBLISH_ROUTE_KEYWORDS = ("发布图文笔记", "发布笔记", "发布到小红书", "小红书发布")


@dataclass
class PublishWorkflowContext:
    messages: list
    history: list[dict]
    workflow: dict
    user_message: str
    note: object | None = None
    article_title: str = ""


@dataclass
class PublishWorkflowNode:
    name: str
    handler: Callable[[PublishWorkflowContext], str]

    def run(self, context: PublishWorkflowContext) -> str:
        return self.handler(context)


def messages_to_history(messages: list) -> list[dict]:
    history: list[dict] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = str(msg.content).strip()
            if content:
                history.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage) and str(msg.content).strip():
            history.append({"role": "assistant", "content": str(msg.content).strip()})
    return history


def last_user_message(messages: list) -> str:
    return next(
        (str(msg.content).strip() for msg in reversed(messages) if isinstance(msg, HumanMessage)),
        "",
    )


def is_publish_intent(text: str) -> bool:
    if not text:
        return False
    if any(keyword in text for keyword in PUBLISH_ROUTE_KEYWORDS):
        return True
    return "发布" in text and ("笔记" in text or "小红书" in text or "图文" in text)


def is_cancel_message(text: str) -> bool:
    return any(keyword in text for keyword in ("取消发布", "先别发", "不要发布", "不发布了", "停止发布"))


def is_continue_message(text: str) -> bool:
    raw = str(text or "").strip()
    keywords = ("满意", "进入下一步", "继续下一步", "继续做封面", "开始做封面", "做封面")
    return any(keyword in raw for keyword in keywords)


def extract_explicit_title(text: str) -> str:
    match = re.search(r"(?:标题|题目)\s*[：:]\s*(.+)", text)
    if not match:
        return ""
    return match.group(1).strip()


def is_yes_message(text: str) -> bool:
    raw = str(text or "").strip()
    return any(keyword in raw for keyword in ("要", "需要", "做", "进行", "是", "yes", "用风格迁移"))


def is_no_message(text: str) -> bool:
    raw = str(text or "").strip()
    return any(keyword in raw for keyword in ("不要", "不需要", "不用", "不做", "否", "no"))


def is_confirm_message(text: str) -> bool:
    keywords = ("确认发布", "可以发布", "确认", "发布吧", "发吧", "开始发布", "确定发布")
    return any(keyword in str(text or "") for keyword in keywords)
