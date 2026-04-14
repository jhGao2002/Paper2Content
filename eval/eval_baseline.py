from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness, context_precision, context_recall


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import MultiAgentApp  # noqa: E402


RESULT_DIR = ROOT / "result"
DATASET_PATH = RESULT_DIR / "eval_dataset.json"
OUTPUT_CSV = RESULT_DIR / "experiment_results_log.csv"


def append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    print("启动基线评测...")
    setup_app = MultiAgentApp(user_id="eval_user", session_id="eval_setup")

    pdf_files = [
        "S-Lora.pdf",
        "UniMoE.pdf",
        "MoLA.pdf",
        "MoRAL.pdf",
        "MoA.pdf",
        "Alpha_lora.pdf",
        "2309.05444v1.pdf",
        "2403.03432v1.pdf",
        "AdapterFusion.pdf",
        "Combining Modular Skills in Multitask Learning.pdf",
        "LoraHub.pdf",
    ]

    print("尝试加载评测文档...")
    for pdf in pdf_files:
        result = setup_app.load_document(str(ROOT / pdf))
        if not result["success"]:
            print(f"加载失败: {pdf} - {result['message']}")

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    questions = []
    ground_truths = []
    answers = []
    contexts = []

    print(f"开始生成答案，共 {len(qa_pairs)} 条样本...")
    for idx, pair in enumerate(qa_pairs, start=1):
        question = pair["question"]
        ground_truth = pair["ground_truth"]

        app = MultiAgentApp(user_id="eval_user", session_id=f"eval_thread_{idx - 1}")
        answer = app.ask(question)

        questions.append(question)
        ground_truths.append(ground_truth)
        answers.append(answer)
        contexts.append([app.last_retrieved_docs])
        print(f"已完成 {idx}/{len(qa_pairs)}")

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    print("开始执行 Ragas 评测...")
    eval_result = evaluate(
        dataset,
        metrics=[context_precision, context_recall, answer_correctness],
        raise_exceptions=False,
    )

    df = eval_result.to_pandas()
    df["experiment_name"] = "Baseline_Naive_RAG"
    df["chunk_size"] = 800
    df["chunk_overlap"] = 150
    df["embedding_model"] = "text-embedding-3-small"
    append_csv(df, OUTPUT_CSV)

    print(f"评测结果已追加到: {OUTPUT_CSV}")
    print(eval_result)


if __name__ == "__main__":
    main()
