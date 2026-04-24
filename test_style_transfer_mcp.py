from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from style.gallery import get_style_image_path
from xhs.style_transfer_service import style_transfer_sync


DEFAULT_CONTENT_PATH = r"G:\AICoding\paper_assitant\result\xhs_note_images\xhs_cover_1776427204_0.png"


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="测试风格迁移 MCP 服务调用效果。")
    parser.add_argument(
        "--content-path",
        default=DEFAULT_CONTENT_PATH,
        help="内容图路径，默认使用已生成好的封面图。",
    )
    parser.add_argument(
        "--style-path",
        default="",
        help="自定义风格图绝对路径；如果不传，则调用服务端默认风格图。",
    )
    parser.add_argument(
        "--style-name",
        default="",
        help="已保存到本地风格图库中的文件名；如果提供，将自动解析为 style_path。",
    )
    args = parser.parse_args()

    content_path = Path(args.content_path).expanduser().resolve()
    if not content_path.exists():
        print(f"[Test] 内容图不存在：{content_path}")
        return 1

    style_path = args.style_path.strip()
    if not style_path and args.style_name.strip():
        resolved = get_style_image_path(args.style_name.strip())
        if not resolved:
            print(f"[Test] 未找到风格图库图片：{args.style_name.strip()}")
            return 1
        style_path = resolved

    print("[Test] 开始调用风格迁移 MCP 服务")
    print(f"[Test] content_path={content_path}")
    print(f"[Test] style_path={style_path or '默认风格图'}")

    try:
        result = style_transfer_sync(
            content_path=str(content_path),
            style_path=style_path,
        )
    except Exception as exc:
        print(f"[Test] 调用失败：{exc}")
        return 1

    print("[Test] 调用结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    cached_content_path = str(result.get("cached_content_path", "")).strip()
    cached_style_path = str(result.get("cached_style_path", "")).strip()
    resolved_content_path = str(result.get("resolved_content_path", "")).strip()
    resolved_style_path = str(result.get("resolved_style_path", "")).strip()
    remote_output_path = str(result.get("remote_output_path", "")).strip()
    local_output_path = str(result.get("local_output_path", "")).strip()

    if cached_content_path:
        print(f"[Test] 内容图已上传到 .cache/content：{cached_content_path}")
    if style_path:
        print(
            f"[Test] 风格图{'已上传到 .cache/style' if cached_style_path else '未上传成功到 .cache/style'}："
            f"{cached_style_path or style_path}"
        )
    else:
        print("[Test] 未提供风格图，服务端将使用默认风格图")

    if resolved_content_path:
        print(f"[Test] 风格迁移实际使用的服务端内容图路径：{resolved_content_path}")
    if resolved_style_path:
        print(f"[Test] 风格迁移实际使用的服务端风格图路径：{resolved_style_path}")
    elif result.get("used_default_style"):
        print("[Test] 风格迁移实际使用的是服务端默认风格图")

    if remote_output_path:
        print(f"[Test] 服务端结果图路径：{remote_output_path}")
    if local_output_path:
        print(f"[Test] 本地缓存结果图路径：{local_output_path}")

    output_path = str(result.get("output_path", "")).strip()
    if result.get("success") and output_path:
        print(f"[Test] 生成结果图：{output_path}")
        return 0

    print("[Test] 风格迁移未成功，请检查 MCP 服务日志。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
