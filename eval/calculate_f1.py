from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result"
INPUT_CSV = RESULT_DIR / "experiment_results_log_v2.csv"
OUTPUT_CSV = RESULT_DIR / "experiment_results_with_f1.csv"


def compute_f1(precision: float, recall: float) -> float:
    if pd.isna(precision) or pd.isna(recall) or (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def main() -> None:
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"找不到结果文件: {INPUT_CSV}")
        return

    required_cols = ["context_precision", "context_recall", "answer_correctness"]
    for col in required_cols:
        if col not in df.columns:
            print(f"缺少必要列: {col}")
            return

    df["retrieval_f1"] = df.apply(
        lambda row: compute_f1(row["context_precision"], row["context_recall"]),
        axis=1,
    )
    df.to_csv(OUTPUT_CSV, index=False)

    mean_precision = df["context_precision"].mean()
    mean_recall = df["context_recall"].mean()
    mean_f1 = df["retrieval_f1"].mean()
    mean_correctness = df["answer_correctness"].mean()

    print("=" * 40)
    print("评测指标汇总")
    print("=" * 40)
    print(f"1. Context Precision:  {mean_precision:.4f}")
    print(f"2. Context Recall:     {mean_recall:.4f}")
    print(f"3. Retrieval F1:       {mean_f1:.4f}")
    print("-" * 40)
    print(f"4. Answer Correctness: {mean_correctness:.4f}")
    print("=" * 40)
    print(f"已写出带 F1 的结果文件: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
