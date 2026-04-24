from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class MCPToolSpec:
    name: str
    description: str = ""
    input_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class StyleTransferToolset:
    upload_content: MCPToolSpec
    style_transfer: MCPToolSpec
    download_generated: MCPToolSpec
    upload_style: MCPToolSpec | None = None


def _endpoint() -> str:
    return os.getenv("STYLE_TRANSFER_MCP_ENDPOINT", "http://127.0.0.1:1234/mcp").strip()


async def _parse_tool_result(result) -> dict[str, Any]:
    if result.content:
        first = result.content[0]
        if hasattr(first, "text"):
            raw = first.text
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"raw": raw}
    return {}


async def _discover_tools(session) -> list[MCPToolSpec]:
    tools_result = await session.list_tools()
    specs: list[MCPToolSpec] = []
    for tool in tools_result.tools:
        specs.append(
            MCPToolSpec(
                name=str(getattr(tool, "name", "")).strip(),
                description=str(getattr(tool, "description", "") or "").strip(),
                input_schema=getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None),
            )
        )
    return [spec for spec in specs if spec.name]


def _tool_payloads(tools: list[MCPToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema or {},
        }
        for tool in tools
    ]


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"LLM 未返回 JSON 对象：{raw}")
    return json.loads(text[start : end + 1])


def _tool_by_name(tools: list[MCPToolSpec]) -> dict[str, MCPToolSpec]:
    return {tool.name: tool for tool in tools}


def _validate_plan(payload: dict[str, Any], tools: list[MCPToolSpec], has_local_style: bool) -> StyleTransferToolset:
    names = _tool_by_name(tools)

    upload_content_name = str(payload.get("upload_content_tool", "")).strip()
    transfer_name = str(payload.get("style_transfer_tool", "")).strip()
    download_name = str(payload.get("download_generated_tool", "")).strip()
    upload_style_value = payload.get("upload_style_tool")
    upload_style_name = str(upload_style_value or "").strip()
    use_custom_style = bool(payload.get("use_custom_style")) and has_local_style

    missing = [
        name
        for name in (upload_content_name, transfer_name, download_name)
        if name not in names
    ]
    if missing:
        raise ValueError(f"LLM 计划引用了不存在的必需工具：{', '.join(missing)}")
    if use_custom_style and upload_style_name not in names:
        raise ValueError("LLM 计划需要自定义风格图，但未给出有效的风格图上传工具。")

    return StyleTransferToolset(
        upload_content=names[upload_content_name],
        upload_style=names[upload_style_name] if use_custom_style else None,
        style_transfer=names[transfer_name],
        download_generated=names[download_name],
    )


def _plan_toolset_with_llm(llm, tools: list[MCPToolSpec], has_local_style: bool) -> StyleTransferToolset:
    prompt = (
        "你是 MCP 工具调用规划器。请根据远程 MCP list_tools 返回的工具能力，"
        "为一次封面图风格迁移生成工具调用计划。\n"
        "任务约束：\n"
        "1. 必须上传内容图，也就是刚生成的小红书封面图。\n"
        "2. 必须调用风格迁移工具。\n"
        "3. 必须下载风格化后的结果图。\n"
        "4. 如果 has_local_style_image 为 true，你可以决定是否上传自定义风格图；"
        "如果不上传，风格迁移的 style_path 参数会传 null。\n"
        "5. 工具名必须完全来自 tools 列表，不要虚构工具。\n\n"
        "只输出 JSON：\n"
        "{\n"
        '  "upload_content_tool": "工具名",\n'
        '  "upload_style_tool": "工具名或null",\n'
        '  "use_custom_style": true/false,\n'
        '  "style_transfer_tool": "工具名",\n'
        '  "download_generated_tool": "工具名"\n'
        "}\n\n"
        f"has_local_style_image: {has_local_style}\n"
        f"tools: {json.dumps(_tool_payloads(tools), ensure_ascii=False)}"
    )
    raw = str(llm.invoke(prompt).content).strip()
    return _validate_plan(_extract_json_object(raw), tools, has_local_style=has_local_style)


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


async def _call_tool(session, tool: MCPToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    _log_progress(f"调用 MCP 工具：{tool.name}")
    result = await session.call_tool(tool.name, arguments=arguments)
    return await _parse_tool_result(result)


async def _upload_image(session, tool: MCPToolSpec, local_path: str, cache_label: str) -> str:
    file_name, image_base64 = _read_image_as_base64(local_path)
    _log_progress(f"上传{cache_label}到服务端缓存，local_path={local_path}")
    result = await _call_tool(
        session,
        tool,
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


async def _download_generated_image(
    session,
    tool: MCPToolSpec,
    remote_image_path: str,
    local_content_path: str,
) -> tuple[str, str]:
    _log_progress(f"下载风格迁移结果图，remote_image_path={remote_image_path}")
    result = await _call_tool(
        session,
        tool,
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


async def style_transfer(content_path: str, style_path: str | None = None, llm=None) -> dict[str, Any]:
    if not content_path:
        raise ValueError("content_path 不能为空。")

    selected_style_path = str(style_path or "").strip()
    need_style_upload = bool(selected_style_path)
    _log_progress(
        f"开始执行风格迁移，content_path={content_path}，style_path={selected_style_path or 'MCP 默认风格'}"
    )

    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    endpoint = _endpoint()
    _log_progress(f"连接风格迁移 MCP 服务，endpoint={endpoint}")
    try:
        async with streamablehttp_client(endpoint) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await _discover_tools(session)
                if llm is None:
                    raise RuntimeError("未提供 LLM，无法基于 list_tools 生成 MCP 工具调用计划。")
                toolset = _plan_toolset_with_llm(llm, tools, has_local_style=need_style_upload)
                _log_progress("LLM 已基于 list_tools 生成 MCP 调用计划")
                _log_progress(
                    "MCP 工具发现完成："
                    f"content={toolset.upload_content.name}, "
                    f"style={toolset.upload_style.name if toolset.upload_style else 'skip'}, "
                    f"transfer={toolset.style_transfer.name}, "
                    f"download={toolset.download_generated.name}"
                )

                server_content_path = await _upload_image(
                    session=session,
                    tool=toolset.upload_content,
                    local_path=content_path,
                    cache_label="内容图",
                )

                server_style_path = ""
                if toolset.upload_style is not None:
                    server_style_path = await _upload_image(
                        session=session,
                        tool=toolset.upload_style,
                        local_path=selected_style_path,
                        cache_label="风格图",
                    )

                arguments = {
                    "content_path": server_content_path,
                    "style_path": server_style_path or None,
                }
                _log_progress(
                    f"调用风格迁移工具，server_content_path={server_content_path}，"
                    f"server_style_path={server_style_path or 'None/MCP 默认风格'}"
                )
                result = await _call_tool(session, toolset.style_transfer, arguments)
                remote_output_path = str(result.get("output_path", "")).strip()
                local_output_path = ""
                downloaded_file_name = ""
                if bool(result.get("success")) and remote_output_path:
                    local_output_path, downloaded_file_name = await _download_generated_image(
                        session=session,
                        tool=toolset.download_generated,
                        remote_image_path=remote_output_path,
                        local_content_path=content_path,
                    )
    except Exception as exc:
        message = _format_exception_message(exc)
        raise RuntimeError(f"风格迁移 MCP 调用失败（endpoint={endpoint}）：{message}") from exc

    return {
        "success": bool(result.get("success")),
        "output_path": local_output_path or remote_output_path,
        "remote_output_path": remote_output_path,
        "local_output_path": local_output_path,
        "downloaded_file_name": downloaded_file_name,
        "used_default_style": not bool(server_style_path) or bool(result.get("used_default_style")),
        "cached_content_path": server_content_path,
        "cached_style_path": server_style_path,
        "resolved_content_path": str(result.get("resolved_content_path", "")).strip(),
        "resolved_style_path": str(result.get("resolved_style_path", "")).strip(),
        "style_path": str(result.get("style_path", "")).strip(),
        "message": str(result.get("message", "")).strip() or str(result.get("raw", "")).strip() or "调用完成",
    }


def style_transfer_sync(content_path: str, style_path: str | None = None, llm=None) -> dict[str, Any]:
    return asyncio.run(style_transfer(content_path=content_path, style_path=style_path, llm=llm))
