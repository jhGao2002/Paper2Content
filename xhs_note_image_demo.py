from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

from xhs.publish_service import (
    check_login_status_sync,
    list_mcp_tools_sync,
    publish_note_with_fallback_sync,
)
from xhs.schemas import XHSPreparedUploadPayload


DEMO_IMAGE_PATH = Path(r"G:\AICoding\paper_assitant\result\xhs_note_images\xhs_cover_1776347991_0.png")
MCP_BINARY_PATH = Path(
    os.getenv(
        "XHS_MCP_BINARY",
        r"G:\AICoding\xiaohongshu-mcp-windows-amd64\xiaohongshu-mcp-windows-amd64.exe",
    )
)
MCP_HOST = os.getenv("XHS_MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("XHS_MCP_PORT", "18060"))


def build_demo_payload() -> XHSPreparedUploadPayload:
    title = "RAGST：把RAG带进风格迁移"
    content = (
        "最近在看风格迁移方向时，我最有感触的一点是：参考图不是越多越好，选错参考图反而会把迁移结果带偏。\n\n"
        "这篇工作真正解决的问题，不是“怎么再堆一个更复杂的迁移模型”，而是先回答“哪张参考图最值得被用”。"
        "它把参考选择这一步自动化后，迁移结果的稳定性和细节质量都会更好。\n\n"
        "如果你也在做参考图驱动的生成或迁移任务，这个思路很值得借鉴：先把检索和筛选做对，再谈后面的生成质量。"
        "\n\n你会更想把这套思路用在风格迁移，还是别的图像生成任务里？"
    )
    return XHSPreparedUploadPayload(
        title=title,
        content=content,
        images=[str(DEMO_IMAGE_PATH)],
        tags=["风格迁移", "RAG", "论文解读", "AIGC", "图像生成"],
        paper_title="RAG for Style Transfer",
        paper_title_short="RAGST",
        is_original=True,
        visibility="公开可见",
    )


def ensure_demo_image() -> None:
    if not DEMO_IMAGE_PATH.exists():
        raise FileNotFoundError(f"演示图片不存在：{DEMO_IMAGE_PATH}")


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_mcp_service(output_dir: Path) -> dict[str, object]:
    if _port_open(MCP_HOST, MCP_PORT):
        print(f"[Demo] 检测到 MCP 服务已在 {MCP_HOST}:{MCP_PORT} 运行", flush=True)
        return {"started": False, "binary": str(MCP_BINARY_PATH), "host": MCP_HOST, "port": MCP_PORT}

    if not MCP_BINARY_PATH.exists():
        raise FileNotFoundError(f"未找到小红书 MCP 可执行文件：{MCP_BINARY_PATH}")

    log_path = output_dir / "xhs_mcp_start.log"
    print(f"[Demo] MCP 服务未启动，准备拉起：{MCP_BINARY_PATH}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        subprocess.Popen(
            [str(MCP_BINARY_PATH)],
            cwd=str(MCP_BINARY_PATH.parent),
            stdout=log_file,
            stderr=log_file,
            creationflags=creationflags,
        )

    deadline = time.time() + 25
    while time.time() < deadline:
        if _port_open(MCP_HOST, MCP_PORT):
            print(f"[Demo] MCP 服务已启动：{MCP_HOST}:{MCP_PORT}", flush=True)
            return {
                "started": True,
                "binary": str(MCP_BINARY_PATH),
                "host": MCP_HOST,
                "port": MCP_PORT,
                "log_path": str(log_path.resolve()),
            }
        time.sleep(1)

    raise RuntimeError(
        f"已尝试启动小红书 MCP，但 {MCP_HOST}:{MCP_PORT} 在 25 秒内未就绪。请查看日志：{log_path.resolve()}"
    )


def main() -> None:
    output_dir = Path("result") / "xhs_note_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_demo_image()
    mcp_bootstrap = ensure_mcp_service(output_dir)
    payload = build_demo_payload()

    login_status: dict[str, object] = {}
    tools: list[str] = []

    print("[Demo] 开始检查 MCP 登录状态...", flush=True)
    try:
        login_status = check_login_status_sync()
        print(f"[Demo] 登录状态：{json.dumps(login_status, ensure_ascii=False)}", flush=True)
    except Exception as exc:
        login_status = {"success": False, "message": str(exc)}
        print(f"[Demo] 登录状态检查失败：{exc}", flush=True)

    print("[Demo] 开始拉取 MCP 工具列表...", flush=True)
    try:
        tools = list_mcp_tools_sync()
        print(f"[Demo] MCP 工具：{tools}", flush=True)
        if "publish_content" not in tools:
            raise RuntimeError(f"MCP 服务未注册 publish_content，当前工具列表：{tools}")
    except Exception as exc:
        tools = []
        print(f"[Demo] MCP 工具检查失败：{exc}", flush=True)

    print("[Demo] 开始自动发布小红书图文...", flush=True)
    result = publish_note_with_fallback_sync(
        payload,
        max_attempts=3,
        retry_delay_seconds=5.0,
    )

    artifact = {
        "payload": payload.to_dict(),
        "mcp_bootstrap": mcp_bootstrap,
        "login_status": login_status,
        "tools": tools,
        "publish_result": result,
    }
    (output_dir / "demo_publish_result.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[Demo] 上传标题：{payload.title}", flush=True)
    print(f"[Demo] 图片路径：{payload.images[0]}", flush=True)
    print(f"[Demo] 发布结果：{json.dumps(result, ensure_ascii=False)}", flush=True)
    print(f"[Demo] 结果已写入：{(output_dir / 'demo_publish_result.json').resolve()}", flush=True)

    if not result.get("success"):
        raise RuntimeError(f"自动发布失败：{result.get('message', '未知错误')}")


if __name__ == "__main__":
    main()
