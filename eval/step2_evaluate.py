from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import answer_correctness, context_precision, context_recall
from ragas.run_config import RunConfig


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result"
INPUT_PATH = RESULT_DIR / "generated_answers_cache_v2.json"
OUTPUT_CSV = RESULT_DIR / "experiment_results_log_v2.csv"


def append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    print("=== Step 2: 评测缓存结果 ===")
    load_dotenv(ROOT / ".env")

    try:
        with INPUT_PATH.open("r", encoding="utf-8") as f:
            saved_results = json.load(f)
    except FileNotFoundError:
        print(f"找不到 {INPUT_PATH}，请先运行 step1_generate.py。")
        return

    print(f"读取到 {len(saved_results)} 条缓存结果。")
    dataset = Dataset.from_dict(
        {
            "question": [item["question"] for item in saved_results],
            "answer": [item["answer"] for item in saved_results],
            "contexts": [item["contexts"] for item in saved_results],
            "ground_truth": [item["ground_truth"] for item in saved_results],
        }
    )

    print("初始化评测模型...")
    judge_embeddings = OpenAIEmbeddings(
        model="Embedding-3",
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_URL"),
        check_embedding_ctx_length=False,
    )
    judge_llm = ChatOpenAI(
        model="glm-4-flash",
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_URL"),
        temperature=1.0,
        max_tokens=65536,
    )

    safe_config = RunConfig(
        max_workers=3,
        max_retries=10,
        max_wait=60,
        timeout=120,
    )

    print("开始执行 Ragas 评测...")
    eval_result = evaluate(
        dataset,
        metrics=[context_precision, context_recall, answer_correctness],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=safe_config,
        raise_exceptions=False,
    )

    df = eval_result.to_pandas()
    df["experiment_name"] = "New_RAG"
    append_csv(df, OUTPUT_CSV)

    print(f"评测结果已追加到: {OUTPUT_CSV}")
    print(eval_result)


if __name__ == "__main__":
    main()
