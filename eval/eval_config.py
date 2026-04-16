from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
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
    embedding_model: str


VARIANTS = [
    VariantConfig(
        slug="01_fixed_chunk_embedding3",
        display_name="Fixed Chunk + Embedding-3",
        use_parent_child=False,
        embedding_model="Embedding-3",
    ),
    VariantConfig(
        slug="02_parent_child_lexical",
        display_name="Parent-Child Chunk + Lexical Embedding",
        use_parent_child=True,
        embedding_model="LexicalHashEmbedding",
    ),
    VariantConfig(
        slug="03_parent_child_embedding3",
        display_name="Parent-Child Chunk + Embedding-3",
        use_parent_child=True,
        embedding_model="Embedding-3",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行三组 RAG 评测变体，并输出中间结果、指标对比和 token 消耗。"
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=PAPERS_DIR,
        help="待评测论文 PDF 目录，默认是 eval/papers_example。",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=SOURCE_DATASET_PATH,
        help="评测集 JSON 路径，默认是 result/eval_dataset.json。",
    )
    parser.add_argument(
        "--questions-per-paper",
        type=int,
        default=5,
        help="正式评测时每篇论文生成的问答数量，默认 5。",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="从论文开头截取给 LLM 生成评测集的最大字符数，默认 12000。",
    )
    parser.add_argument(
        "--regenerate-dataset",
        action="store_true",
        help="忽略已有评测集，重新生成 eval_dataset.json。",
    )
    parser.add_argument(
        "--beta",
        action="store_true",
        help="只读取第一篇论文，生成极小规模样本做冒烟测试。",
    )
    parser.add_argument(
        "--beta-question-limit",
        type=int,
        default=2,
        help="beta 模式最终保留的问答数量，默认 2。",
    )
    return parser.parse_args()
