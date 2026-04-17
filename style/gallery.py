from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path


STYLE_IMAGE_ROOT = Path(os.getenv("STYLE_IMAGE_LIBRARY_DIR", "style_gallery"))
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _ensure_root() -> Path:
    STYLE_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    return STYLE_IMAGE_ROOT


def _is_allowed_image(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def _unique_target_path(filename: str) -> Path:
    root = _ensure_root()
    candidate = root / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        next_candidate = root / f"{stem}_{index}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        index += 1


def list_style_images() -> list[dict]:
    root = _ensure_root()
    items: list[dict] = []
    for path in root.iterdir():
        if not path.is_file() or not _is_allowed_image(path):
            continue
        stat = path.stat()
        items.append(
            {
                "filename": path.name,
                "path": str(path.resolve()),
                "uploaded_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )
    return sorted(items, key=lambda item: item["uploaded_at"], reverse=True)


def save_style_image(uploaded_path: str) -> dict:
    source = Path(uploaded_path)
    if not source.exists():
        return {"success": False, "message": f"文件不存在：{uploaded_path}"}
    if not _is_allowed_image(source):
        return {"success": False, "message": "仅支持上传 png/jpg/jpeg/webp/bmp 图片。"}

    target = _unique_target_path(source.name)
    shutil.copy2(source, target)
    return {
        "success": True,
        "message": f"已保存风格图：{target.name}",
        "filename": target.name,
        "path": str(target.resolve()),
    }


def delete_style_image(filename: str) -> bool:
    target = _ensure_root() / filename
    if not target.exists() or not target.is_file():
        return False
    target.unlink()
    return True


def get_style_image_path(filename: str) -> str | None:
    if not filename:
        return None
    target = _ensure_root() / filename
    if not target.exists() or not target.is_file():
        return None
    return str(target.resolve())
