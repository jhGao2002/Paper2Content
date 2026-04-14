from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import MultiAgentApp  # noqa: E402


RESULT_DIR = ROOT / "result"
DATASET_PATH = RESULT_DIR / "eval_dataset.json"
OUTPUT_PATH = RESULT_DIR / "generated_answers_cache_v2.json"


def main() -> None:
    print("=== Step 1: 生成回答缓存 ===")

    test_user = "eval_user"
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    results_to_save = []
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"开始处理 {len(qa_pairs)} 条评测样本...")
    for idx, pair in enumerate(qa_pairs, start=1):
        question = pair["question"]
        ground_truth = pair["ground_truth"]

        try:
            app = MultiAgentApp(user_id=test_user, session_id=f"eval_thread_{idx - 1}")
            answer = app.ask(question)
            results_to_save.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": answer,
                    "contexts": [app.last_retrieved_docs],
                }
            )

            with OUTPUT_PATH.open("w", encoding="utf-8") as f:
                json.dump(results_to_save, f, ensure_ascii=False, indent=2)

            print(f"已完成 {idx}/{len(qa_pairs)}")
        except Exception as exc:
            print(f"第 {idx} 条生成失败: {exc}")

    print(f"缓存已写入: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
