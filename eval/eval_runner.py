from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain_core.callbacks import get_usage_metadata_callback

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
)

from ragas import evaluate
from ragas.metrics import answer_correctness, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from config import get_embeddings, get_fast_llm, get_llm
from generate_eval_dataset import generate_dataset_from_papers
from memory.store import FaissVectorStore

from eval_pipelines import build_embeddings, build_pipeline
from eval_utils import compute_f1, load_json, normalize_usage, save_json, write_dataframe


def select_pdf_files(papers_dir: Path, beta: bool) -> list[Path]:
    pdf_files = sorted(path for path in papers_dir.glob("*.pdf") if path.is_file())
    if not pdf_files:
        raise FileNotFoundError(f"未在目录中找到 PDF 文件：{papers_dir}")
    return pdf_files[:1] if beta else pdf_files


def build_dataset_path(dataset_path: Path, run_dir: Path, beta: bool) -> Path:
    if beta:
        return run_dir / "dataset" / "beta_eval_dataset.json"
    return dataset_path


def ensure_eval_dataset(
    dataset_path: Path,
    papers_dir: Path,
    run_dir: Path,
    pdf_files: list[Path],
    questions_per_paper: int,
    max_chars: int,
    regenerate_dataset: bool,
    beta: bool,
    beta_question_limit: int,
) -> tuple[list[dict], dict[str, int], Path]:
    target_path = build_dataset_path(dataset_path, run_dir, beta)
    usage_summary = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    need_generate = regenerate_dataset or not target_path.exists() or beta
    if need_generate:
        target_papers_dir = papers_dir
        if beta:
            beta_papers_dir = run_dir / "dataset" / "beta_papers"
            beta_papers_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf_files[0], beta_papers_dir / pdf_files[0].name)
            target_papers_dir = beta_papers_dir

        with get_usage_metadata_callback() as cb:
            generate_dataset_from_papers(
                papers_dir=target_papers_dir,
                output_path=target_path,
                questions_per_paper=beta_question_limit if beta else questions_per_paper,
                max_chars=max_chars,
            )
        usage_summary = normalize_usage(cb.usage_metadata)

    dataset = load_json(target_path)
    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f"评测集为空或格式不正确：{target_path}")

    if beta:
        dataset = dataset[:beta_question_limit]
        save_json(target_path, dataset)

    snapshot_path = run_dir / "dataset" / "eval_dataset.json"
    save_json(snapshot_path, dataset)
    return dataset, usage_summary, snapshot_path


def get_variant_dir(run_dir: Path, variant_slug: str, timestamp: str) -> Path:
    variant_dir = run_dir / f"{variant_slug}_{timestamp}"
    for path in [variant_dir / "answers", variant_dir / "metrics", variant_dir / "runtime"]:
        path.mkdir(parents=True, exist_ok=True)
    return variant_dir


def run_variant(
    variant,
    qa_pairs: list[dict],
    pdf_files: list[Path],
    run_dir: Path,
    timestamp: str,
) -> tuple[list[dict], dict[str, int], dict, Path]:
    variant_dir = get_variant_dir(run_dir, variant.slug, timestamp)
    answers_path = variant_dir / "answers" / "answers.json"
    embeddings, embedding_name = build_embeddings(variant.embedding_model)
    metadata = {
        "variant": variant.slug,
        "display_name": variant.display_name,
        "chunking_strategy": "parent_child" if variant.use_parent_child else "fixed_flat",
        "embedding": embedding_name,
        "parent_chunk_size": 1500 if variant.use_parent_child else None,
        "parent_chunk_overlap": 200 if variant.use_parent_child else None,
        "child_chunk_size": 400 if variant.use_parent_child else None,
        "child_chunk_overlap": 50 if variant.use_parent_child else None,
        "flat_chunk_size": None if variant.use_parent_child else 800,
        "flat_chunk_overlap": None if variant.use_parent_child else 150,
        "child_retrieval_top_k": 8 if variant.use_parent_child else None,
        "final_context_top_k": 4,
        "variant_dir": str(variant_dir.relative_to(run_dir)),
    }

    if answers_path.exists():
        print(f"=== {variant.display_name}: 复用已有回答缓存 ===")
        return (
            load_json(answers_path),
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            metadata,
            variant_dir,
        )

    store = FaissVectorStore(
        embedding=embeddings,
        collection_name="pdf_knowledge",
        persist_root=variant_dir / "runtime" / "vectorstores",
    )
    llm = get_llm()
    pipeline = build_pipeline(variant.use_parent_child, store, llm)

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
    save_json(answers_path, records)
    return records, usage_summary, metadata, variant_dir


def evaluate_variant(
    variant,
    records: list[dict],
    metadata: dict,
    generation_usage: dict[str, int],
    variant_dir: Path,
    run_dir: Path,
) -> tuple[dict, dict[str, int]]:
    metrics_path = variant_dir / "metrics" / "metrics.csv"
    if metrics_path.exists():
        print(f"=== {variant.display_name}: 复用已有评测结果 ===")
        df = pd.read_csv(metrics_path)
        judge_usage = {
            "input_tokens": int(df["judge_input_tokens"].iloc[0]) if "judge_input_tokens" in df.columns and not df.empty else 0,
            "output_tokens": int(df["judge_output_tokens"].iloc[0]) if "judge_output_tokens" in df.columns and not df.empty else 0,
            "total_tokens": int(df["judge_total_tokens"].iloc[0]) if "judge_total_tokens" in df.columns and not df.empty else 0,
        }
        generation_usage = {
            "input_tokens": int(df["generation_input_tokens"].iloc[0]) if "generation_input_tokens" in df.columns and not df.empty else generation_usage["input_tokens"],
            "output_tokens": int(df["generation_output_tokens"].iloc[0]) if "generation_output_tokens" in df.columns and not df.empty else generation_usage["output_tokens"],
            "total_tokens": int(df["generation_total_tokens"].iloc[0]) if "generation_total_tokens" in df.columns and not df.empty else generation_usage["total_tokens"],
        }
        return _build_summary(variant, df, metadata, generation_usage, judge_usage, metrics_path, run_dir), judge_usage

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
            metrics=[context_precision, context_recall, answer_correctness, faithfulness],
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
    write_dataframe(df, metrics_path)

    return _build_summary(variant, df, metadata, generation_usage, judge_usage, metrics_path, run_dir), judge_usage


def _build_summary(
    variant,
    df: pd.DataFrame,
    metadata: dict,
    generation_usage: dict[str, int],
    judge_usage: dict[str, int],
    metrics_path: Path,
    run_dir: Path,
) -> dict:
    return {
        "variant": variant.slug,
        "display_name": variant.display_name,
        "sample_count": len(df),
        "context_precision": float(df["context_precision"].mean()),
        "context_recall": float(df["context_recall"].mean()),
        "retrieval_f1": float(df["retrieval_f1"].mean()),
        "answer_correctness": float(df["answer_correctness"].mean()),
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
