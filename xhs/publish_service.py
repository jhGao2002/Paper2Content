from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any
from urllib import error, request

from xhs.schemas import XHSMCPPublishArgs, XHSNoteDraft, XHSPreparedUploadPayload


def _log_progress(message: str) -> None:
    print(f"[XHSPublishService] {message}", flush=True)


def _format_exception_message(exc: Exception) -> str:
    parts: list[str] = []

    def _walk(error: BaseException) -> None:
        text = str(error).strip()
        label = error.__class__.__name__
        parts.append(f"{label}: {text}" if text else label)
        nested = getattr(error, "exceptions", None)
        if nested:
            for item in nested:
                if isinstance(item, BaseException):
                    _walk(item)

    _walk(exc)
    return " | ".join(dict.fromkeys(parts))


def _clean_tags(tags: list[str]) -> list[str]:
    clean: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        value = tag.lstrip("#").strip()
        if not value or value in seen:
            continue
        clean.append(value)
        seen.add(value)
        if len(clean) >= 10:
            break
    return clean


def build_mcp_publish_args(
    note: XHSNoteDraft,
    image_paths: list[str],
    is_original: bool = True,
    visibility: str = "公开可见",
) -> XHSMCPPublishArgs:
    content = note.body.strip()
    if note.cta and note.cta not in content:
        content = f"{content}\n\n{note.cta}".strip()

    return XHSMCPPublishArgs(
        title=note.title[:20],
        content=content,
        images=list(image_paths),
        tags=_clean_tags(note.hashtags),
        is_original=is_original,
        visibility=visibility,
    )


def build_mcp_publish_args_from_payload(payload: XHSPreparedUploadPayload) -> XHSMCPPublishArgs:
    return XHSMCPPublishArgs(
        title=payload.title[:20],
        content=payload.content.strip(),
        images=list(payload.images),
        tags=_clean_tags(payload.tags),
        is_original=payload.is_original,
        visibility=payload.visibility,
    )


def build_rest_publish_payload(payload: XHSPreparedUploadPayload) -> dict[str, Any]:
    return {
        "Title": payload.title[:20],
        "Content": payload.content.strip(),
        "Images": list(payload.images),
        "Tags": _clean_tags(payload.tags),
        "IsOriginal": payload.is_original,
        "Visibility": payload.visibility,
    }


async def _call_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    endpoint = os.getenv("XHS_MCP_ENDPOINT", "http://localhost:18060/mcp").strip()
    _log_progress(f"连接 MCP 服务，endpoint={endpoint}")
    try:
        async with streamablehttp_client(endpoint) as (read, write, _):
            async with ClientSession(read, write) as session:
                _log_progress("MCP session initialize")
                await session.initialize()
                _log_progress(f"调用 MCP 工具：{tool_name}")
                result = await session.call_tool(tool_name, arguments=arguments)
                _log_progress(f"MCP 工具调用完成：{tool_name}")
    except Exception as exc:
        message = _format_exception_message(exc)
        raise RuntimeError(f"MCP 服务调用失败（tool={tool_name}, endpoint={endpoint}）：{message}") from exc

    if result.content:
        first = result.content[0]
        if hasattr(first, "text"):
            raw = first.text
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"raw": raw}
    return {}


async def check_login_status() -> dict[str, Any]:
    _log_progress("检查小红书登录状态")
    return await _call_tool("check_login_status", {})


async def list_mcp_tools() -> list[str]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    endpoint = os.getenv("XHS_MCP_ENDPOINT", "http://localhost:18060/mcp").strip()
    _log_progress(f"列出 MCP 工具，endpoint={endpoint}")
    try:
        async with streamablehttp_client(endpoint) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
    except Exception as exc:
        message = _format_exception_message(exc)
        raise RuntimeError(f"MCP 工具列表获取失败（endpoint={endpoint}）：{message}") from exc
    return [tool.name for tool in tools_result.tools]


async def publish_note_via_mcp(args: XHSMCPPublishArgs) -> dict[str, Any]:
    if not args.images:
        raise ValueError("发布到小红书前至少需要 1 张图片。")

    _log_progress(
        f"开始发布，title={args.title}，images={len(args.images)}，tags={len(args.tags)}，visibility={args.visibility}"
    )
    tool_args: dict[str, Any] = {
        "title": args.title,
        "content": args.content,
        "images": args.images,
        "is_original": args.is_original,
        "visibility": args.visibility,
    }
    if args.tags:
        tool_args["tags"] = args.tags

    result = await _call_tool("publish_content", tool_args)
    raw_text = str(result.get("raw", ""))
    has_error = result.get("success") is False or "error" in raw_text.lower() or "失败" in raw_text
    _log_progress(f"发布结果已返回，success={not has_error if raw_text else result.get('success', True)}")
    return {
        "success": not has_error if raw_text else result.get("success", True),
        "message": result.get("message") or raw_text or "发布成功",
        "data": result.get("data"),
        "mode": "mcp",
    }


def publish_note_via_mcp_sync(args: XHSMCPPublishArgs) -> dict[str, Any]:
    return asyncio.run(publish_note_via_mcp(args))


def check_login_status_sync() -> dict[str, Any]:
    return asyncio.run(check_login_status())


def list_mcp_tools_sync() -> list[str]:
    return asyncio.run(list_mcp_tools())


def publish_note_via_rest_sync(payload: XHSPreparedUploadPayload) -> dict[str, Any]:
    base_url = os.getenv("XHS_MCP_URL", "http://localhost:18060").strip().rstrip("/")
    url = f"{base_url}/api/v1/publish"
    body = build_rest_publish_payload(payload)
    _log_progress(f"开始调用 MCP REST 发布接口，url={url}")
    req = request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return {
            "success": False,
            "message": f"REST API 返回 HTTP {exc.code}: {detail}",
            "mode": "rest",
        }
    except Exception as exc:
        return {
            "success": False,
            "message": f"REST 调用异常: {_format_exception_message(exc)}",
            "mode": "rest",
        }

    return {
        "success": bool(result.get("success")),
        "message": str(result.get("message", "")).strip() or "发布完成",
        "data": result.get("data"),
        "mode": "rest",
    }


def publish_note_with_retry_sync(
    args: XHSMCPPublishArgs,
    max_attempts: int = 3,
    retry_delay_seconds: float = 3.0,
) -> dict[str, Any]:
    last_result: dict[str, Any] | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        _log_progress(f"发布尝试 {attempt}/{max_attempts}")
        try:
            result = publish_note_via_mcp_sync(args)
        except Exception as exc:
            result = {
                "success": False,
                "message": f"MCP 调用异常: {exc}",
                "mode": "mcp",
            }
        last_result = result
        if result.get("success"):
            return result

        message = str(result.get("message", "")).strip()
        _log_progress(f"第 {attempt} 次发布失败：{message or '未知错误'}")
        if attempt >= max_attempts:
            break
        retryable_markers = (
            "ERR_CONNECTION_CLOSED",
            "没有找到发布 TAB",
            "导航到发布页面失败",
            "上传图文",
            "502",
            "Bad Gateway",
            "timeout",
            "超时",
        )
        if not any(marker in message for marker in retryable_markers):
            break
        _log_progress(f"等待 {retry_delay_seconds:.1f}s 后重试")
        time.sleep(retry_delay_seconds)
    return last_result or {"success": False, "message": "发布失败", "mode": "mcp"}


def publish_note_with_fallback_sync(
    payload: XHSPreparedUploadPayload,
    max_attempts: int = 3,
    retry_delay_seconds: float = 3.0,
) -> dict[str, Any]:
    mcp_args = build_mcp_publish_args_from_payload(payload)
    direct_result = publish_note_with_retry_sync(
        mcp_args,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    if direct_result.get("success"):
        direct_result["path"] = "mcp"
        return direct_result

    _log_progress("MCP 协议发布失败，开始回退到同一服务的 REST 发布接口")
    rest_result = publish_note_via_rest_sync(payload)
    rest_result["previous_error"] = direct_result
    rest_result["path"] = "rest_fallback"
    return rest_result
