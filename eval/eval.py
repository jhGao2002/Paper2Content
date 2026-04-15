from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
import pymupdf4llm
from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.documents import Document
from ragas import evaluate
from ragas.metrics.collections import (
    answer_correctness,
    context_precision,
    context_recall,
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
RESULT_DIR = ROOT / "result"
RUNTIME_ROOT = RESULT_DIR / "eval_runtime"
README_PATH = EVAL_DIR / "README.md"
DATASET_PATH = RESULT_DIR / "eval_dataset.json"
FULL_CACHE_PATH = RESULT_DIR / "generated_answers_cache_v2.json"
BASELINE_CACHE_PATH = RESULT_DIR / "generated_answers_cache_baseline.json"
FULL_CSV_PATH = RESULT_DIR / "experiment_results_log_v2.csv"
BASELINE_CSV_PATH = RESULT_DIR / "experiment_results_log_baseline.csv"
FULL_F1_PATH = RESULT_DIR / "experiment_results_with_f1.csv"
BASELINE_F1_PATH = RESULT_DIR / "experiment_results_with_f1_baseline.csv"
SUMMARY_JSON_PATH = RESULT_DIR / "eval_comparison_summary.json"
SUMMARY_CSV_PATH = RESULT_DIR / "eval_comparison_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行完整评测：生成数据集、baseline 对比、当前系统评测。")
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=PAPERS_DIR,
        help="评测论文目录，默认是 eval/papers_example",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="评测数据集路径，默认是 result/eval_dataset.json",
    )
    parser.add_argument(
        "--questions-per-paper",
        type=int,
        default=8,
        help="当需要自动生成评测集时，每篇论文生成的问题数量",
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
        help="强制重新生成 eval_dataset.json",
    )
    return parser.parse_args()


def compute_f1(precision: float, recall: float) -> float:
    if pd.isna(precision) or pd.isna(recall) or (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def load_eval_dataset(dataset_path: Path) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"评测集为空或格式不正确: {dataset_path}")
    return data


def ensure_eval_dataset(args: argparse.Namespace) -> list[dict]:
    dataset_path = args.dataset.resolve()
    papers_dir = args.papers_dir.resolve()
    if args.regenerate_dataset or not dataset_path.exists():
        print("未找到评测集或已指定重新生成，开始基于论文自动构造 eval_dataset.json ...")
        generate_dataset_from_papers(
            papers_dir=papers_dir,
            output_path=dataset_path,
            questions_per_paper=args.questions_per_paper,
            max_chars=args.max_chars,
        )
    return load_eval_dataset(dataset_path)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class NaiveRAGBaseline:
    def __init__(self, embeddings, llm, persist_root: Path):
        self.store = FaissVectorStore(
            embedding=embeddings,
            collection_name="baseline_pdf_knowledge",
            persist_root=persist_root,
        )
        self.llm = llm

    def load_document(self, pdf_path: Path) -> dict:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        chunks = split_text(md_text, chunk_size=800, chunk_overlap=150)
        docs = []
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk.page_content,
                    metadata={"source": pdf_path.name},
                )
            )
        self.store.add_documents(docs)
        return {"success": True, "document": pdf_path.name, "chunk_count": len(docs)}

    def ask(self, question: str) -> tuple[str, str]:
        docs = self.store.similarity_search(question, k=4)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = (
            "你是一个朴素 RAG 问答助手。请仅根据给定检索片段回答问题。\n"
            "如果片段中没有足够证据，请明确说信息不足，不要编造。\n\n"
            f"问题：{question}\n\n"
            f"检索片段：\n{context}"
        )
        answer = self.llm.invoke(prompt).content
        return str(answer).strip(), context


@contextmanager
def isolated_full_runtime(runtime_dir: Path):
    import document.registry as registry_module
    import memory.store as memory_store_module
    import session.manager as session_manager_module

    original_docs_file = registry_module.DOCS_FILE
    original_vector_root = memory_store_module.VECTOR_ROOT
    original_sessions_file = session_manager_module.SESSIONS_FILE
    original_sessions_db = session_manager_module.SESSIONS_DB

    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    registry_module.DOCS_FILE = str(runtime_dir / "documents.json")
    memory_store_module.VECTOR_ROOT = runtime_dir / "vectorstores"
    session_manager_module.SESSIONS_FILE = str(runtime_dir / "sessions.json")
    session_manager_module.SESSIONS_DB = str(runtime_dir / "sessions.db")
    try:
        yield
    finally:
        registry_module.DOCS_FILE = original_docs_file
        memory_store_module.VECTOR_ROOT = original_vector_root
        session_manager_module.SESSIONS_FILE = original_sessions_file
        session_manager_module.SESSIONS_DB = original_sessions_db


def run_baseline_eval(qa_pairs: list[dict], pdf_files: list[Path]) -> list[dict]:
    runtime_dir = RUNTIME_ROOT / "baseline"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()
    llm = get_llm()
    app = NaiveRAGBaseline(embeddings=embeddings, llm=llm, persist_root=runtime_dir / "vectorstores")

    print("=== Baseline: 载入论文 ===")
    for pdf_path in pdf_files:
        result = app.load_document(pdf_path)
        print(f"  已载入 {result['document']}，chunk 数: {result['chunk_count']}")

    print("=== Baseline: 生成回答 ===")
    records = []
    total = len(qa_pairs)
    for idx, pair in enumerate(qa_pairs, start=1):
        answer, context = app.ask(pair["question"])
        records.append(
            {
                "question": pair["question"],
                "ground_truth": pair["ground_truth"],
                "answer": answer,
                "contexts": [context],
                "source": pair.get("source", ""),
            }
        )
        print(f"  Baseline 已完成 {idx}/{total}")
    save_json(BASELINE_CACHE_PATH, records)
    return records


def run_full_eval(qa_pairs: list[dict], pdf_files: list[Path]) -> list[dict]:
    runtime_dir = RUNTIME_ROOT / "full"
    with isolated_full_runtime(runtime_dir):
        from main import MultiAgentApp

        print("=== Full System: 载入论文 ===")
        setup_app = MultiAgentApp(user_id="eval_full_user", session_id="eval_full_setup")
        for pdf_path in pdf_files:
            result = setup_app.load_document(str(pdf_path))
            if not result["success"]:
                raise RuntimeError(f"载入失败: {pdf_path.name} - {result['message']}")
            print(f"  已载入 {pdf_path.name}")

        print("=== Full System: 生成回答 ===")
        records = []
        total = len(qa_pairs)
        for idx, pair in enumerate(qa_pairs, start=1):
            app = MultiAgentApp(
                user_id="eval_full_user",
                session_id=f"eval_full_thread_{idx}",
            )
            answer = app.ask(pair["question"])
            records.append(
                {
                    "question": pair["question"],
                    "ground_truth": pair["ground_truth"],
                    "answer": answer,
                    "contexts": [app.last_retrieved_docs],
                    "source": pair.get("source", ""),
                }
            )
            print(f"  Full System 已完成 {idx}/{total}")

    save_json(FULL_CACHE_PATH, records)
    return records


def evaluate_records(records: list[dict], experiment_name: str, output_csv: Path, output_f1_csv: Path, metadata: dict) -> dict:
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
    safe_config = RunConfig(
        max_workers=3,
        max_retries=10,
        max_wait=60,
        timeout=120,
    )

    eval_result = evaluate(
        dataset,
        metrics=[context_precision, context_recall, answer_correctness],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=safe_config,
        raise_exceptions=False,
    )

    df = eval_result.to_pandas()
    df["retrieval_f1"] = df.apply(
        lambda row: compute_f1(row["context_precision"], row["context_recall"]),
        axis=1,
    )
    df["experiment_name"] = experiment_name
    for key, value in metadata.items():
        df[key] = value

    write_dataframe(df, output_csv)
    write_dataframe(df, output_f1_csv)

    summary = {
        "experiment_name": experiment_name,
        "sample_count": len(records),
        "context_precision": float(df["context_precision"].mean()),
        "context_recall": float(df["context_recall"].mean()),
        "retrieval_f1": float(df["retrieval_f1"].mean()),
        "answer_correctness": float(df["answer_correctness"].mean()),
        "metadata": metadata,
    }
    return summary


def compare_summaries(baseline_summary: dict, full_summary: dict) -> list[dict]:
    metrics = ["context_precision", "context_recall", "retrieval_f1", "answer_correctness"]
    comparison = []
    for metric in metrics:
        comparison.append(
            {
                "metric": metric,
                "baseline": baseline_summary[metric],
                "full_system": full_summary[metric],
                "delta": full_summary[metric] - baseline_summary[metric],
            }
        )
    return comparison


def append_results_to_readme(
    baseline_summary: dict,
    full_summary: dict,
    comparison_rows: list[dict],
    dataset_size: int,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "",
        f"### {timestamp}",
        f"- 样本数：{dataset_size}",
        f"- Baseline（Naive RAG）：answer_correctness={baseline_summary['answer_correctness']:.4f}，context_precision={baseline_summary['context_precision']:.4f}，context_recall={baseline_summary['context_recall']:.4f}，retrieval_f1={baseline_summary['retrieval_f1']:.4f}",
        f"- Full System（当前项目）：answer_correctness={full_summary['answer_correctness']:.4f}，context_precision={full_summary['context_precision']:.4f}，context_recall={full_summary['context_recall']:.4f}，retrieval_f1={full_summary['retrieval_f1']:.4f}",
        "- 指标差值（Full - Baseline）：",
    ]
    for row in comparison_rows:
        lines.append(
            f"  - {row['metric']}: {row['delta']:+.4f}"
        )

    with README_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    dataset = ensure_eval_dataset(args)
    pdf_files = sorted(path for path in args.papers_dir.resolve().glob("*.pdf") if path.is_file())
    if not pdf_files:
        raise FileNotFoundError(f"未在论文目录中找到 PDF: {args.papers_dir.resolve()}")

    baseline_records = run_baseline_eval(dataset, pdf_files)
    full_records = run_full_eval(dataset, pdf_files)

    baseline_summary = evaluate_records(
        baseline_records,
        experiment_name="Baseline_Naive_RAG",
        output_csv=BASELINE_CSV_PATH,
        output_f1_csv=BASELINE_F1_PATH,
        metadata={
            "chunk_size": 800,
            "chunk_overlap": 150,
            "retrieval_top_k": 4,
            "query_expansion": 0,
            "rerank": False,
            "parent_child_chunking": False,
            "plan_and_act": False,
            "memory": False,
        },
    )
    full_summary = evaluate_records(
        full_records,
        experiment_name="Current_Project_Full",
        output_csv=FULL_CSV_PATH,
        output_f1_csv=FULL_F1_PATH,
        metadata={
            "parent_chunk_size": 1500,
            "parent_chunk_overlap": 200,
            "child_chunk_size": 400,
            "child_chunk_overlap": 50,
            "child_top_k_per_query": 3,
            "query_expansion": 3,
            "rerank": True,
            "parent_child_chunking": True,
            "plan_and_act": True,
            "memory": True,
        },
    )

    comparison_rows = compare_summaries(baseline_summary, full_summary)
    save_json(
        SUMMARY_JSON_PATH,
        {
            "baseline": baseline_summary,
            "full_system": full_summary,
            "comparison": comparison_rows,
            "generated_at": datetime.now().isoformat(timespec="minutes"),
        },
    )
    write_dataframe(pd.DataFrame(comparison_rows), SUMMARY_CSV_PATH)
    append_results_to_readme(baseline_summary, full_summary, comparison_rows, len(dataset))

    print("=" * 48)
    print("评测完成")
    print("=" * 48)
    print(f"Baseline 结果: {BASELINE_CSV_PATH}")
    print(f"Full 结果: {FULL_CSV_PATH}")
    print(f"对比摘要: {SUMMARY_JSON_PATH}")
    print(f"README 已追加结果记录: {README_PATH}")


if __name__ == "__main__":
    main()
