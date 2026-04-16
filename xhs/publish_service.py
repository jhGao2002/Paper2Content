from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from xhs.schemas import XHSMCPPublishArgs, XHSNoteDraft


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


async def _call_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    endpoint = os.getenv("XHS_MCP_ENDPOINT", "http://localhost:18060/mcp").strip()
    async with streamablehttp_client(endpoint) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)

    if result.content:
        first = result.content[0]
        if hasattr(first, "text"):
            raw = first.text
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"raw": raw}
    return {}


async def publish_note_via_mcp(args: XHSMCPPublishArgs) -> dict[str, Any]:
    if not args.images:
        raise ValueError("发布到小红书前至少需要 1 张图片。")

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
    return {
        "success": not has_error if raw_text else result.get("success", True),
        "message": result.get("message") or raw_text or "发布成功",
        "data": result.get("data"),
        "mode": "mcp",
    }


def publish_note_via_mcp_sync(args: XHSMCPPublishArgs) -> dict[str, Any]:
    return asyncio.run(publish_note_via_mcp(args))
