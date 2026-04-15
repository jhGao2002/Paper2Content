# document/loader.py
# 负责 PDF 文档的解析和向量入库。
# 使用父子块策略：父块（1500 字）保留完整上下文，子块（400 字）用于精细检索。
# 入库完成后调用 fast_llm 提取论文标题和摘要，注册到 documents.json 持久化。
import os
import time
import uuid

import pymupdf4llm

from document.chunking import split_text
from document.registry import is_registered, register_document


def load_document(pdf_path: str, pdf_store, user_id: str, fast_llm=None) -> dict:
    """解析 PDF 并以父子块策略入库，返回操作结果字典。"""
    if not os.path.exists(pdf_path):
        return {"success": False, "message": f"文件不存在: {pdf_path}"}

    filename = os.path.basename(pdf_path)
    start_time = time.time()
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        parent_docs = split_text(md_text, chunk_size=1500, chunk_overlap=200)
        child_docs = []
        for p_doc in parent_docs:
            p_id = str(uuid.uuid4())
            p_text = p_doc.page_content
            for c in split_text(p_text, chunk_size=400, chunk_overlap=50):
                c.metadata = {
                    "parent_id": p_id,
                    "parent_text": p_text,
                    "source": filename,
                    "user_id": user_id,
                }
                child_docs.append(c)

        pdf_store.add_documents(child_docs)
        elapsed = time.time() - start_time

        if fast_llm and not is_registered(filename):
            _register_metadata(filename, md_text, len(child_docs), fast_llm)
        elif not fast_llm and not is_registered(filename):
            register_document(
                filename,
                title=filename,
                summary="（未生成摘要）",
                chunk_count=len(child_docs),
            )

        return {
            "success": True,
            "message": f"解析成功 (耗时: {elapsed:.1f}s)，入库 {len(child_docs)} 个子块",
            "document": filename,
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


def _register_metadata(filename: str, md_text: str, chunk_count: int, fast_llm):
    """用 fast_llm 从文档开头提取标题和生成摘要。"""
    excerpt = md_text[:3000]
    prompt = (
        "请从以下学术论文内容中提取信息，并严格按下面格式输出，不要添加任何说明：\n"
        "[TITLE]\n"
        f"<论文完整标题，如无法确定则输出文件名 {filename}>\n"
        "[SUMMARY]\n"
        "<用 2-3 句话概括论文的研究问题、方法和主要贡献，可分成多行>\n\n"
        f"论文内容：\n{excerpt}"
    )
    try:
        raw = fast_llm.invoke(prompt).content
        if isinstance(raw, list):
            raw = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw
            )
        raw = str(raw).strip()

        title = filename
        summary = "（未能提取摘要）"

        if "[TITLE]" in raw and "[SUMMARY]" in raw:
            title_block, summary_block = raw.split("[SUMMARY]", 1)
            title = title_block.split("[TITLE]", 1)[-1].strip() or filename
            summary = summary_block.strip() or "（未能提取摘要）"
        else:
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            title_line = next((line for line in lines if line.startswith("标题")), "")
            if title_line:
                title = title_line.split("：", 1)[-1].strip() or filename

            summary_start = next(
                (idx for idx, line in enumerate(lines) if line.startswith("摘要")),
                -1,
            )
            if summary_start != -1:
                first_line = lines[summary_start].split("：", 1)[-1].strip()
                tail_lines = []
                for line in lines[summary_start + 1:]:
                    if line.startswith("标题"):
                        break
                    tail_lines.append(line)
                summary_text = "\n".join(
                    part for part in [first_line, *tail_lines] if part
                ).strip()
                if summary_text:
                    summary = summary_text
    except Exception:
        title, summary = filename, "（未能提取摘要）"

    register_document(filename, title=title, summary=summary, chunk_count=chunk_count)
