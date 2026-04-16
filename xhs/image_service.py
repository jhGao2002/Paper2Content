from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib import error, request

from xhs.schemas import XHSNoteDraft


def _image_api_key() -> str:
    return (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_IMAGE_API_KEY")
        or ""
    ).strip()


def _image_api_url() -> str:
    return (
        os.getenv("DASHSCOPE_IMAGE_ENDPOINT")
        or "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    ).strip()


def _image_model() -> str:
    return os.getenv("DASHSCOPE_IMAGE_MODEL", "z-image-turbo").strip() or "z-image-turbo"


def _image_size() -> str:
    size = os.getenv("DASHSCOPE_IMAGE_SIZE", "512*512").strip() or "512*512"
    return size.replace("x", "*").replace("X", "*")


def _prompt_extend_enabled() -> bool:
    raw = os.getenv("DASHSCOPE_IMAGE_PROMPT_EXTEND", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _image_output_dir() -> Path:
    raw = os.getenv("XHS_NOTE_IMAGE_OUTPUT_DIR", "result/xhs_note_images").strip() or "result/xhs_note_images"
    return Path(raw)


def build_cover_prompt(note: XHSNoteDraft) -> str:
    image_plan = note.image_plan
    props = "、".join(image_plan.props) if image_plan.props else "纸张便签、电脑界面卡片、重点标记"
    style = "、".join(image_plan.style_keywords) if image_plan.style_keywords else "轻写实、明亮、干净、适合小红书封面"
    scene = image_plan.scene or "安静的桌面工作场景"
    main_subject = image_plan.main_subject or "一位正在整理知识点的年轻创作者"
    pain_point = image_plan.pain_point_visual or f"混乱、卡住、反复试错，围绕问题“{note.core_problem}”"
    solution = image_plan.solution_visual or f"清晰、被整理好的结果，突出“{note.solved_problem}”"
    composition = image_plan.composition or "方形封面构图，主体居中，前后景有明显层次，对比明确"
    palette = image_plan.color_palette or "奶白、浅米、暖橙、浅青"

    return (
        "请生成一张适合小红书图文封面的 1:1 配图。"
        f"主题是“{note.solved_problem or note.core_problem}”。"
        f"主体：{main_subject}。"
        f"场景：{scene}。"
        f"画面中的问题状态：{pain_point}。"
        f"画面中的解决状态：{solution}。"
        f"道具元素：{props}。"
        f"构图：{composition}。"
        f"色彩：{palette}。"
        f"风格关键词：{style}。"
        "请重点通过人物动作、物件状态对比、信息被整理后的秩序感，来表现这篇内容解决了什么问题。"
        "不要依赖大段中文标题，不要满版海报字，不要做成纯文字封面。"
        "可以出现极少量无法辨认的小标签或界面元素，但不能让文字成为画面主体。"
        "整体要求清晰、生活化、有知识分享感，适合 512x512 正方形封面。"
    )


def build_cover_negative_prompt(note: XHSNoteDraft) -> str:
    avoid = list(note.image_plan.avoid_elements)
    avoid.extend(
        [
            "大段中文",
            "满版海报字",
            "水印",
            "logo",
            "低清晰度",
            "人物手部畸形",
            "拥挤杂乱背景",
            "纯抽象图标拼贴",
        ]
    )
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in avoid:
        value = item.strip()
        if value and value not in seen:
            cleaned.append(value)
            seen.add(value)
    return "，".join(cleaned)


def _post_json(url: str, payload: dict, api_key: str) -> dict:
    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        if "模型不存在" in detail or "model does not exist" in detail.lower():
            model = str(payload.get("model", "")).strip() or "unknown"
            raise RuntimeError(
                f"图片模型 {model} 当前不可用。"
                "如果你需要继续验证链路，请在 .env 里调整 DASHSCOPE_IMAGE_MODEL 为当前账号已开通的模型。"
            ) from exc
        raise RuntimeError(f"图片接口返回 HTTP {exc.code}: {detail}") from exc


def _download_file(url: str, target_path: Path) -> None:
    req = request.Request(url, headers={"User-Agent": "paper_assistant/1.0"})
    with request.urlopen(req, timeout=120) as resp:
        target_path.write_bytes(resp.read())


def generate_cover_images(
    note: XHSNoteDraft,
    image_count: int = 1,
    output_dir: Path | None = None,
) -> list[str]:
    api_key = _image_api_key()
    if not api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY（可回退使用 QWEN_IMAGE_API_KEY）。")

    prompt = build_cover_prompt(note)
    output_root = output_dir or _image_output_dir()
    output_root.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for index in range(max(1, image_count)):
        payload = {
            "model": _image_model(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": prompt,
                            }
                        ],
                    }
                ]
            },
            "parameters": {
                "prompt_extend": _prompt_extend_enabled(),
                "size": _image_size(),
            },
        }
        response = _post_json(_image_api_url(), payload, api_key)
        if response.get("code") and response.get("message"):
            raise RuntimeError(f"图片接口调用失败：{response.get('code')} - {response.get('message')}")

        choices = (((response.get("output") or {}).get("choices")) or [])
        message = ((choices[0] or {}).get("message")) if choices else {}
        content_items = (message or {}).get("content") or []
        image_url = ""
        for item in content_items:
            image_url = str((item or {}).get("image", "")).strip()
            if image_url:
                break
        if not image_url:
            raise RuntimeError(f"图片接口未返回 image 字段: {response}")

        target_path = output_root / f"xhs_cover_{int(time.time())}_{index}.png"
        _download_file(image_url, target_path)
        saved_paths.append(str(target_path.resolve()))

    return saved_paths
