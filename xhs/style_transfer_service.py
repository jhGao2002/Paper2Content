from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any


def _log_progress(message: str) -> None:
    print(f"[XHSStyleTransfer] {message}", flush=True)


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


async def _call_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    endpoint = os.getenv("STYLE_TRANSFER_MCP_ENDPOINT", "http://127.0.0.1:1234/mcp").strip()
    _log_progress(f"连接风格迁移 MCP 服务，endpoint={endpoint}")
    try:
        async with streamablehttp_client(endpoint) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                _log_progress(f"调用 MCP 工具：{tool_name}")
                result = await session.call_tool(tool_name, arguments=arguments)
    except Exception as exc:
        message = _format_exception_message(exc)
        raise RuntimeError(f"风格迁移 MCP 调用失败（tool={tool_name}, endpoint={endpoint}）：{message}") from exc

    if result.content:
        first = result.content[0]
        if hasattr(first, "text"):
            raw = first.text
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"raw": raw}
    return {}


def _read_image_as_base64(local_path: str) -> tuple[str, str]:
    path = Path(local_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"本地图片不存在：{path}")

    return path.name, base64.b64encode(path.read_bytes()).decode("ascii")


def _local_result_path(content_path: str, remote_file_name: str) -> Path:
    source = Path(content_path).expanduser().resolve()
    output_dir = source.parent
    suffix = Path(remote_file_name).suffix or source.suffix or ".png"
    base_name = f"{source.stem}_styled"
    candidate = output_dir / f"{base_name}{suffix}"
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        next_candidate = output_dir / f"{base_name}_{index}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        index += 1


async def _upload_image(local_path: str, tool_name: str, cache_label: str) -> str:
    file_name, image_base64 = _read_image_as_base64(local_path)
    _log_progress(f"上传{cache_label}到服务端缓存，local_path={local_path}")
    result = await _call_tool(
        tool_name,
        {
            "file_name": file_name,
            "image_base64": image_base64,
        },
    )
    if not result.get("success"):
        message = str(result.get("message", "")).strip() or "服务端未返回成功状态。"
        raise RuntimeError(f"{cache_label}上传失败：{message}")

    cached_path = str(result.get("cached_path", "")).strip()
    if not cached_path:
        raise RuntimeError(f"{cache_label}上传失败：服务端未返回 cached_path。")

    _log_progress(f"{cache_label}上传成功，cached_path={cached_path}")
    return cached_path


async def _download_generated_image(remote_image_path: str, local_content_path: str) -> tuple[str, str]:
    _log_progress(f"下载风格迁移结果图，remote_image_path={remote_image_path}")
    result = await _call_tool(
        "download_generated_image_tool",
        {
            "image_path": remote_image_path,
        },
    )
    if not result.get("success"):
        message = str(result.get("message", "")).strip() or "服务端未返回成功状态。"
        raise RuntimeError(f"结果图下载失败：{message}")

    image_base64 = str(result.get("image_base64", "")).strip()
    if not image_base64:
        raise RuntimeError("结果图下载失败：服务端未返回 image_base64。")

    file_name = str(result.get("file_name", "")).strip() or Path(remote_image_path).name or "styled_output.png"
    local_path = _local_result_path(local_content_path, file_name)
    local_path.write_bytes(base64.b64decode(image_base64))
    _log_progress(f"结果图已缓存到本地，local_path={local_path}")
    return str(local_path.resolve()), file_name


async def style_transfer(content_path: str, style_path: str = "") -> dict[str, Any]:
    if not content_path:
        raise ValueError("content_path 不能为空。")

    _log_progress(
        f"开始执行风格迁移，content_path={content_path}，style_path={style_path or '默认风格'}"
    )

    server_content_path = await _upload_image(
        local_path=content_path,
        tool_name="upload_content_image_tool",
        cache_label="内容图",
    )

    server_style_path = ""
    if style_path:
        server_style_path = await _upload_image(
            local_path=style_path,
            tool_name="upload_style_image_tool",
            cache_label="风格图",
        )

    arguments = {"content_path": server_content_path}
    if server_style_path:
        arguments["style_path"] = server_style_path

    _log_progress(
        f"调用风格迁移工具，server_content_path={server_content_path}，"
        f"server_style_path={server_style_path or '默认风格'}"
    )
    result = await _call_tool("style_transfer_tool", arguments)
    remote_output_path = str(result.get("output_path", "")).strip()
    local_output_path = ""
    downloaded_file_name = ""
    if bool(result.get("success")) and remote_output_path:
        local_output_path, downloaded_file_name = await _download_generated_image(
            remote_image_path=remote_output_path,
            local_content_path=content_path,
        )

    return {
        "success": bool(result.get("success")),
        "output_path": local_output_path or remote_output_path,
        "remote_output_path": remote_output_path,
        "local_output_path": local_output_path,
        "downloaded_file_name": downloaded_file_name,
        "used_default_style": bool(result.get("used_default_style")),
        "cached_content_path": server_content_path,
        "cached_style_path": server_style_path,
        "resolved_content_path": str(result.get("resolved_content_path", "")).strip(),
        "resolved_style_path": str(result.get("resolved_style_path", "")).strip(),
        "style_path": str(result.get("style_path", "")).strip(),
        "message": str(result.get("message", "")).strip() or str(result.get("raw", "")).strip() or "调用完成",
    }


def style_transfer_sync(content_path: str, style_path: str = "") -> dict[str, Any]:
    return asyncio.run(style_transfer(content_path=content_path, style_path=style_path))
