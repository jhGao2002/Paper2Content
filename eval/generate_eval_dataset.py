from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pymupdf4llm


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_llm  # noqa: E402


PAPERS_DIR = Path(__file__).resolve().parent / "papers_example"
OUTPUT_PATH = ROOT / "result" / "eval_dataset.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 eval/papers_example 中的论文生成 RAG 评测集。"
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=PAPERS_DIR,
        help="论文 PDF 目录，默认是 eval/papers_example。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="输出 JSON 路径，默认是 result/eval_dataset.json。",
    )
    parser.add_argument(
        "--questions-per-paper",
        type=int,
        default=5,
        help="每篇论文生成的问答数量，默认 5。",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="从论文开头截取给 LLM 的最大字符数，默认 12000。",
    )
    return parser.parse_args()


def collect_pdf_files(papers_dir: Path) -> list[Path]:
    return sorted(path for path in papers_dir.glob("*.pdf") if path.is_file())


def extract_json_block(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM 返回内容中没有找到合法 JSON 数组")
    return text[start : end + 1]


def build_prompt(filename: str, excerpt: str, questions_per_paper: int) -> str:
    return f"""你是学术论文评测集构造助手。请基于论文内容生成适合 RAG 评测的问答样本。

要求：
1. 围绕论文的研究问题、方法、实验设置、关键结论和局限性，生成 {questions_per_paper} 条样本。
2. 输出必须是 JSON 数组，每个元素只包含两个字段：
   - "question"
   - "ground_truth"
3. 不要输出 JSON 之外的解释。
4. 问题要具体、可回答，避免过于宽泛。
5. ground_truth 要尽量忠实于论文内容，长度适中，适合作为评测参考答案。
6. 如果论文开头信息不足，也只能基于提供内容生成，不要编造。

论文文件名：{filename}
论文节选：
{excerpt}
"""


def generate_samples_for_pdf(
    pdf_path: Path,
    llm,
    questions_per_paper: int,
    max_chars: int,
) -> list[dict[str, str]]:
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    excerpt = md_text[:max_chars]
    prompt = build_prompt(pdf_path.name, excerpt, questions_per_paper)
    raw = llm.invoke(prompt).content
    raw_text = str(raw).strip()
    json_text = extract_json_block(raw_text)
    samples = json.loads(json_text)

    cleaned_samples = []
    for item in samples:
        question = str(item.get("question", "")).strip()
        ground_truth = str(item.get("ground_truth", "")).strip()
        if not question or not ground_truth:
            continue
        cleaned_samples.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "source": pdf_path.name,
            }
        )
    return cleaned_samples


def generate_dataset_from_papers(
    papers_dir: Path,
    output_path: Path,
    questions_per_paper: int,
    max_chars: int,
) -> list[dict[str, str]]:
    if not papers_dir.exists():
        raise FileNotFoundError(f"论文目录不存在：{papers_dir}")

    pdf_files = collect_pdf_files(papers_dir)
    if not pdf_files:
        raise FileNotFoundError(f"未在目录中找到 PDF 文件：{papers_dir}")

    llm = get_llm()
    dataset = []

    print(f"开始生成评测集，共 {len(pdf_files)} 篇论文...")
    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{index}/{len(pdf_files)}] 处理 {pdf_path.name}")
        samples = generate_samples_for_pdf(
            pdf_path=pdf_path,
            llm=llm,
            questions_per_paper=questions_per_paper,
            max_chars=max_chars,
        )
        dataset.extend(samples)
        print(f"  生成 {len(samples)} 条样本")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"评测集已写入: {output_path}")
    print(f"总样本数: {len(dataset)}")
    print("建议人工抽查 10% 到 20% 的样本，确认问题和标准答案质量。")
    return dataset


def main() -> None:
    args = parse_args()
    generate_dataset_from_papers(
        papers_dir=args.papers_dir.resolve(),
        output_path=args.output.resolve(),
        questions_per_paper=args.questions_per_paper,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
