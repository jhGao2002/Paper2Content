from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from eval_config import README_PATH, RESULT_ROOT


def create_run_dirs(beta: bool) -> tuple[Path, Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{'beta' if beta else 'full'}_{timestamp}"
    run_dir = RESULT_ROOT / run_name
    latest_dir = RESULT_ROOT / "latest"

    for path in [run_dir / "dataset", run_dir / "summaries"]:
        path.mkdir(parents=True, exist_ok=True)

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, latest_dir, timestamp


def copy_run_to_latest(run_dir: Path, latest_dir: Path) -> None:
    for child in run_dir.iterdir():
        destination = latest_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)


def normalize_usage(usage_metadata: dict) -> dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for payload in usage_metadata.values():
        total["input_tokens"] += int(payload.get("input_tokens", 0))
        total["output_tokens"] += int(payload.get("output_tokens", 0))
        total["total_tokens"] += int(payload.get("total_tokens", 0))
    return total


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def compute_f1(precision: float, recall: float) -> float:
    if pd.isna(precision) or pd.isna(recall) or (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def append_results_to_readme(
    summaries: list[dict],
    dataset_size: int,
    dataset_generation_usage: dict[str, int],
    run_dir: Path,
    beta: bool,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode = "beta" if beta else "full"
    lines = [
        "",
        f"### {timestamp}",
        f"- 模式：`{mode}`",
        f"- 评测样本数：`{dataset_size}`",
        f"- 结果目录：`{run_dir}`",
    ]
    if dataset_generation_usage["total_tokens"] > 0:
        lines.append(
            "- 评测集生成 token："
            f"`input={dataset_generation_usage['input_tokens']}` / "
            f"`output={dataset_generation_usage['output_tokens']}` / "
            f"`total={dataset_generation_usage['total_tokens']}`"
        )

    lines.extend(
        [
            "",
            "| 变体 | Precision | Recall | Retrieval F1 | Correctness | Faithfulness | 回答生成 Tokens | Ragas 评测 Tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in summaries:
        lines.append(
            f"| {summary['variant']} | "
            f"{summary['context_precision']:.4f} | "
            f"{summary['context_recall']:.4f} | "
            f"{summary['retrieval_f1']:.4f} | "
            f"{summary['answer_correctness']:.4f} | "
            f"{summary['faithfulness']:.4f} | "
            f"{summary['generation_total_tokens']} | "
            f"{summary['judge_total_tokens']} |"
        )

    with README_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
