from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf4llm
from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.documents import Document
from ragas import evaluate
from ragas.metrics.collections import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_embeddings, get_fast_llm, get_llm  # noqa: E402
from document.chunking import split_text  # noqa: E402
from generate_eval_dataset import generate_dataset_from_papers  # noqa: E402
from memory.store import FaissVectorStore  # noqa: E402


EVAL_DIR = Path(__file__).resolve().parent
PAPERS_DIR = EVAL_DIR / "papers_example"
README_PATH = EVAL_DIR / "README.md"
RESULT_ROOT = ROOT / "result" / "eval_runs"
SOURCE_DATASET_PATH = ROOT / "result" / "eval_dataset.json"


@dataclass(frozen=True)
class VariantConfig:
    slug: str
    display_name: str
    use_parent_child: bool
    use_embedding3: bool


VARIANTS = [
    VariantConfig(
        slug="01_baseline_flat_lexical",
        display_name="Baseline / Flat Chunk / Lexical Embedding",
        use_parent_child=False,
        use_embedding3=False,
    ),
    VariantConfig(
        slug="02_flat_embedding3",
        display_name="Flat Chunk / Embedding-3",
        use_parent_child=False,
        use_embedding3=True,
    ),
    VariantConfig(
        slug="03_parent_child_lexical",
        display_name="Parent-Child Chunk / Lexical Embedding",
        use_parent_child=True,
        use_embedding3=False,
    ),
    VariantConfig(
        slug="04_parent_child_embedding3",
        display_name="Parent-Child Chunk / Embedding-3",
        use_parent_child=True,
        use_embedding3=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 4 组 RAG 变体评测：分块方式 x embedding 方案。"
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=PAPERS_DIR,
        help="测试论文目录，默认是 eval/papers_example",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=SOURCE_DATASET_PATH,
        help="评测数据集路径，默认是 result/eval_dataset.json",
    )
    parser.add_argument(
        "--questions-per-paper",
        type=int,
        default=8,
        help="如果需要自动生成评测集，每篇论文生成的问题数量",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="自动生成评测集时，每篇论文送入 LLM 的最大字符数",
    )
    parser.add_argument(
        "--regenerate-dataset",
        action="store_true",
        help="强制重新生成评测集",
    )
    return parser.parse_args()


class LexicalHashEmbeddings:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype="float32")
        for token in self._tokenize(text):
            bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vector[bucket] += 1.0
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FlatChunkRAG:
    def __init__(self, store: FaissVectorStore, llm):
        self.store = store
        self.llm = llm

    def load_document(self, pdf_path: Path) -> dict:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        chunks = split_text(md_text, chunk_size=800, chunk_overlap=150)
        docs = [
            Document(page_content=chunk.page_content, metadata={"source": pdf_path.name})
            for chunk in chunks
        ]
        self.store.add_documents(docs)
        return {"document": pdf_path.name, "chunk_count": len(docs)}

    def ask(self, question: str) -> tuple[str, str]:
        docs = self.store.similarity_search(question, k=4)
        context = "\n\n".join(doc.page_content for doc in docs)
        answer = self.llm.invoke(build_answer_prompt(question, context)).content
        return str(answer).strip(), context


class ParentChildRAG:
    def __init__(self, store: FaissVectorStore, llm):
        self.store = store
        self.llm = llm

    def load_document(self, pdf_path: Path) -> dict:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        parent_docs = split_text(md_text, chunk_size=1500, chunk_overlap=200)
        child_docs = []
        for parent_index, parent_doc in enumerate(parent_docs):
            parent_id = f"{pdf_path.name}::parent::{parent_index}"
            parent_text = parent_doc.page_content
            for child_doc in split_text(parent_text, chunk_size=400, chunk_overlap=50):
                child_doc.metadata = {
                    "source": pdf_path.name,
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                }
                child_docs.append(child_doc)
        self.store.add_documents(child_docs)
        return {"document": pdf_path.name, "chunk_count": len(child_docs)}

    def ask(self, question: str) -> tuple[str, str]:
        child_docs = self.store.similarity_search(question, k=8)
        parent_map = {}
        for doc in child_docs:
            parent_id = doc.metadata.get("parent_id", "")
            if parent_id and parent_id not in parent_map:
                parent_map[parent_id] = doc.metadata.get("parent_text", doc.page_content)
            if len(parent_map) >= 4:
                break
        context = "\n\n".join(parent_map.values())
        answer = self.llm.invoke(build_answer_prompt(question, context)).content
        return str(answer).strip(), context


def build_answer_prompt(question: str, context: str) -> str:
    return (
        "你是一个论文问答助手。请严格依据给定检索上下文回答问题。\n"
        "如果上下文证据不足，请明确说信息不足，不要编造。\n\n"
        f"问题：{question}\n\n"
        f"检索上下文：\n{context}"
    )


def compute_f1(precision: float, recall: float) -> float:
    if pd.isna(precision) or pd.isna(recall) or (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def create_run_dirs() -> tuple[Path, Path]:
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = RESULT_ROOT / run_name
    latest_dir = RESULT_ROOT / "latest"
    for path in [
        run_dir / "dataset",
        run_dir / "answers",
        run_dir / "metrics",
        run_dir / "summaries",
        run_dir / "runtime",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, latest_dir


def copy_run_to_latest(run_dir: Path, latest_dir: Path) -> None:
    for child in run_dir.iterdir():
        destination = latest_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)


def normalize_usage(usage_metadata: dict) -> dict[str, int]:
    total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    for payload in usage_metadata.values():
        total["input_tokens"] += int(payload.get("input_tokens", 0))
        total["output_tokens"] += int(payload.get("output_tokens", 0))
        total["total_tokens"] += int(payload.get("total_tokens", 0))
    return total


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def ensure_eval_dataset(args: argparse.Namespace, run_dir: Path) -> tuple[list[dict], dict[str, int]]:
    dataset_path = args.dataset.resolve()
    usage_summary = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    if args.regenerate_dataset or not dataset_path.exists():
        with get_usage_metadata_callback() as cb:
            generate_dataset_from_papers(
                papers_dir=args.papers_dir.resolve(),
                output_path=dataset_path,
                questions_per_paper=args.questions_per_paper,
                max_chars=args.max_chars,
            )
        usage_summary = normalize_usage(cb.usage_metadata)

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f"评测集为空或格式不正确: {dataset_path}")

    save_json(run_dir / "dataset" / "eval_dataset.json", dataset)
    return dataset, usage_summary


def build_embedding(use_embedding3: bool):
    if use_embedding3:
        return get_embeddings(), "Embedding-3"
    return LexicalHashEmbeddings(), "LexicalHashEmbedding"


def build_pipeline(variant: VariantConfig, store: FaissVectorStore, llm):
    if variant.use_parent_child:
        return ParentChildRAG(store, llm)
    return FlatChunkRAG(store, llm)


def run_variant(
    variant: VariantConfig,
    qa_pairs: list[dict],
    pdf_files: list[Path],
    run_dir: Path,
) -> tuple[list[dict], dict[str, int], dict]:
    embeddings, embedding_name = build_embedding(variant.use_embedding3)
    variant_runtime_dir = run_dir / "runtime" / variant.slug
    store = FaissVectorStore(
        embedding=embeddings,
        collection_name="pdf_knowledge",
        persist_root=variant_runtime_dir / "vectorstores",
    )
    llm = get_llm()
    pipeline = build_pipeline(variant, store, llm)

    print(f"=== {variant.display_name}: 加载论文 ===")
    for pdf_path in pdf_files:
        result = pipeline.load_document(pdf_path)
        print(f"  已加载 {result['document']}，chunk 数: {result['chunk_count']}")

    print(f"=== {variant.display_name}: 生成回答 ===")
    with get_usage_metadata_callback() as cb:
        records = []
        total = len(qa_pairs)
        for idx, pair in enumerate(qa_pairs, start=1):
            answer, context = pipeline.ask(pair["question"])
            records.append(
                {
                    "question": pair["question"],
                    "ground_truth": pair["ground_truth"],
                    "answer": answer,
                    "contexts": [context],
                    "source": pair.get("source", ""),
                }
            )
            print(f"  已完成 {idx}/{total}")
    usage_summary = normalize_usage(cb.usage_metadata)

    save_json(run_dir / "answers" / f"{variant.slug}.json", records)
    metadata = {
        "variant": variant.slug,
        "display_name": variant.display_name,
        "chunking_strategy": "parent_child" if variant.use_parent_child else "flat",
        "embedding": embedding_name,
        "parent_chunk_size": 1500 if variant.use_parent_child else None,
        "parent_chunk_overlap": 200 if variant.use_parent_child else None,
        "child_chunk_size": 400 if variant.use_parent_child else None,
        "child_chunk_overlap": 50 if variant.use_parent_child else None,
        "flat_chunk_size": None if variant.use_parent_child else 800,
        "flat_chunk_overlap": None if variant.use_parent_child else 150,
        "retrieval_top_k": 4,
        "query_expansion": False,
        "rerank": False,
    }
    return records, usage_summary, metadata


def evaluate_variant(
    variant: VariantConfig,
    records: list[dict],
    metadata: dict,
    generation_usage: dict[str, int],
    run_dir: Path,
) -> tuple[dict, dict[str, int]]:
    dataset = Dataset.from_dict(
        {
            "question": [item["question"] for item in records],
            "answer": [item["answer"] for item in records],
            "contexts": [item["contexts"] for item in records],
            "ground_truth": [item["ground_truth"] for item in records],
        }
    )

    judge_embeddings = get_embeddings()
    judge_llm = get_fast_llm()
    safe_config = RunConfig(max_workers=3, max_retries=10, max_wait=60, timeout=120)

    with get_usage_metadata_callback() as cb:
        eval_result = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                answer_correctness,
                answer_relevancy,
                faithfulness,
            ],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=safe_config,
            raise_exceptions=False,
        )
    judge_usage = normalize_usage(cb.usage_metadata)

    df = eval_result.to_pandas()
    df["retrieval_f1"] = df.apply(
        lambda row: compute_f1(row["context_precision"], row["context_recall"]),
        axis=1,
    )
    for key, value in metadata.items():
        df[key] = value
    df["generation_input_tokens"] = generation_usage["input_tokens"]
    df["generation_output_tokens"] = generation_usage["output_tokens"]
    df["generation_total_tokens"] = generation_usage["total_tokens"]
    df["judge_input_tokens"] = judge_usage["input_tokens"]
    df["judge_output_tokens"] = judge_usage["output_tokens"]
    df["judge_total_tokens"] = judge_usage["total_tokens"]

    metrics_path = run_dir / "metrics" / f"{variant.slug}.csv"
    write_dataframe(df, metrics_path)

    summary = {
        "variant": variant.slug,
        "display_name": variant.display_name,
        "sample_count": len(records),
        "context_precision": float(df["context_precision"].mean()),
        "context_recall": float(df["context_recall"].mean()),
        "retrieval_f1": float(df["retrieval_f1"].mean()),
        "answer_correctness": float(df["answer_correctness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "faithfulness": float(df["faithfulness"].mean()),
        "generation_input_tokens": generation_usage["input_tokens"],
        "generation_output_tokens": generation_usage["output_tokens"],
        "generation_total_tokens": generation_usage["total_tokens"],
        "judge_input_tokens": judge_usage["input_tokens"],
        "judge_output_tokens": judge_usage["output_tokens"],
        "judge_total_tokens": judge_usage["total_tokens"],
        "metadata": metadata,
        "metrics_file": str(metrics_path.relative_to(run_dir)),
    }
    return summary, judge_usage


def build_comparison_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    metric_names = [
        "context_precision",
        "context_recall",
        "retrieval_f1",
        "answer_correctness",
        "answer_relevancy",
        "faithfulness",
        "generation_total_tokens",
        "judge_total_tokens",
    ]
    for summary in summaries:
        base = {
            "variant": summary["variant"],
            "display_name": summary["display_name"],
        }
        for metric_name in metric_names:
            base[metric_name] = summary[metric_name]
        rows.append(base)
    return rows


def append_results_to_readme(
    summaries: list[dict],
    dataset_size: int,
    dataset_generation_usage: dict[str, int],
    run_dir: Path,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "",
        f"### {timestamp}",
        f"- 样本数：{dataset_size}",
        f"- 结果目录：`{run_dir}`",
    ]
    if dataset_generation_usage["total_tokens"] > 0:
        lines.append(
            "- 评测集自动生成 token："
            f"input={dataset_generation_usage['input_tokens']} / "
            f"output={dataset_generation_usage['output_tokens']} / "
            f"total={dataset_generation_usage['total_tokens']}"
        )

    for summary in summaries:
        lines.append(
            f"- {summary['variant']}："
            f"F1={summary['retrieval_f1']:.4f}，"
            f"Precision={summary['context_precision']:.4f}，"
            f"Recall={summary['context_recall']:.4f}，"
            f"Correctness={summary['answer_correctness']:.4f}，"
            f"Relevancy={summary['answer_relevancy']:.4f}，"
            f"Faithfulness={summary['faithfulness']:.4f}，"
            f"生成 tokens={summary['generation_total_tokens']}，"
            f"评测 tokens={summary['judge_total_tokens']}"
        )

    with README_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    run_dir, latest_dir = create_run_dirs()

    dataset, dataset_generation_usage = ensure_eval_dataset(args, run_dir)
    pdf_files = sorted(args.papers_dir.resolve().glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"未在论文目录中找到 PDF: {args.papers_dir.resolve()}")

    variant_summaries = []
    token_rows = []
    for variant in VARIANTS:
        records, generation_usage, metadata = run_variant(
            variant=variant,
            qa_pairs=dataset,
            pdf_files=pdf_files,
            run_dir=run_dir,
        )
        summary, judge_usage = evaluate_variant(
            variant=variant,
            records=records,
            metadata=metadata,
            generation_usage=generation_usage,
            run_dir=run_dir,
        )
        variant_summaries.append(summary)
        token_rows.append(
            {
                "variant": variant.slug,
                "display_name": variant.display_name,
                "generation_input_tokens": generation_usage["input_tokens"],
                "generation_output_tokens": generation_usage["output_tokens"],
                "generation_total_tokens": generation_usage["total_tokens"],
                "judge_input_tokens": judge_usage["input_tokens"],
                "judge_output_tokens": judge_usage["output_tokens"],
                "judge_total_tokens": judge_usage["total_tokens"],
            }
        )

    summary_df = pd.DataFrame(
        [
            {
                "variant": summary["variant"],
                "display_name": summary["display_name"],
                "sample_count": summary["sample_count"],
                "context_precision": summary["context_precision"],
                "context_recall": summary["context_recall"],
                "retrieval_f1": summary["retrieval_f1"],
                "answer_correctness": summary["answer_correctness"],
                "answer_relevancy": summary["answer_relevancy"],
                "faithfulness": summary["faithfulness"],
                "generation_total_tokens": summary["generation_total_tokens"],
                "judge_total_tokens": summary["judge_total_tokens"],
            }
            for summary in variant_summaries
        ]
    )
    comparison_df = pd.DataFrame(build_comparison_rows(variant_summaries))
    token_df = pd.DataFrame(token_rows)

    write_dataframe(summary_df, run_dir / "summaries" / "variant_metrics_summary.csv")
    write_dataframe(comparison_df, run_dir / "summaries" / "variant_comparison.csv")
    write_dataframe(token_df, run_dir / "summaries" / "variant_token_summary.csv")
    save_json(
        run_dir / "summaries" / "run_manifest.json",
        {
            "generated_at": datetime.now().isoformat(timespec="minutes"),
            "dataset_generation_tokens": dataset_generation_usage,
            "variants": variant_summaries,
        },
    )

    copy_run_to_latest(run_dir, latest_dir)
    append_results_to_readme(
        summaries=variant_summaries,
        dataset_size=len(dataset),
        dataset_generation_usage=dataset_generation_usage,
        run_dir=run_dir,
    )

    print("=" * 60)
    print("评测完成")
    print("=" * 60)
    print(f"本次结果目录: {run_dir}")
    print(f"最新结果目录: {latest_dir}")
    print(f"汇总表: {run_dir / 'summaries' / 'variant_metrics_summary.csv'}")
    print(f"Token 汇总: {run_dir / 'summaries' / 'variant_token_summary.csv'}")


if __name__ == "__main__":
    main()
