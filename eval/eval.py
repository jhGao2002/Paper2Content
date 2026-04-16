from __future__ import annotations

import sys
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from eval_config import ROOT, VARIANTS, parse_args


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_runner import ensure_eval_dataset, evaluate_variant, run_variant, select_pdf_files  # noqa: E402
from eval_utils import append_results_to_readme, copy_run_to_latest, create_run_dirs, save_json, write_dataframe  # noqa: E402


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()

    papers_dir = args.papers_dir.resolve()
    dataset_path = args.dataset.resolve()
    run_dir, latest_dir, timestamp = create_run_dirs(args.beta)
    pdf_files = select_pdf_files(papers_dir, args.beta)
    dataset, dataset_generation_usage, dataset_snapshot_path = ensure_eval_dataset(
        dataset_path=dataset_path,
        papers_dir=papers_dir,
        run_dir=run_dir,
        pdf_files=pdf_files,
        questions_per_paper=args.questions_per_paper,
        max_chars=args.max_chars,
        regenerate_dataset=args.regenerate_dataset,
        beta=args.beta,
        beta_question_limit=args.beta_question_limit,
    )

    print(f"本次加载论文数: {len(pdf_files)}")
    print(f"本次评测集: {dataset_snapshot_path}")

    variant_summaries = []
    token_rows = []
    for variant in VARIANTS:
        records, generation_usage, metadata, variant_dir = run_variant(
            variant=variant,
            qa_pairs=dataset,
            pdf_files=pdf_files,
            run_dir=run_dir,
            timestamp=timestamp,
        )
        summary, _judge_usage = evaluate_variant(
            variant=variant,
            records=records,
            metadata=metadata,
            generation_usage=generation_usage,
            variant_dir=variant_dir,
            run_dir=run_dir,
        )
        variant_summaries.append(summary)
        token_rows.append(
            {
                "variant": variant.slug,
                "display_name": variant.display_name,
                "variant_dir": str(variant_dir.relative_to(run_dir)),
                "generation_input_tokens": summary["generation_input_tokens"],
                "generation_output_tokens": summary["generation_output_tokens"],
                "generation_total_tokens": summary["generation_total_tokens"],
                "judge_input_tokens": summary["judge_input_tokens"],
                "judge_output_tokens": summary["judge_output_tokens"],
                "judge_total_tokens": summary["judge_total_tokens"],
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
                "faithfulness": summary["faithfulness"],
                "generation_total_tokens": summary["generation_total_tokens"],
                "judge_total_tokens": summary["judge_total_tokens"],
                "variant_dir": summary["metadata"]["variant_dir"],
            }
            for summary in variant_summaries
        ]
    )
    token_df = pd.DataFrame(token_rows)
    write_dataframe(summary_df, run_dir / "summaries" / "variant_metrics_summary.csv")
    write_dataframe(token_df, run_dir / "summaries" / "variant_token_summary.csv")
    save_json(
        run_dir / "summaries" / "run_manifest.json",
        {
            "generated_at": datetime.now().isoformat(timespec="minutes"),
            "beta": args.beta,
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
        beta=args.beta,
    )

    print("=" * 60)
    print("评测完成")
    print("=" * 60)
    print(f"结果目录: {run_dir}")
    print(f"latest 快照: {latest_dir}")
    print(f"指标汇总: {run_dir / 'summaries' / 'variant_metrics_summary.csv'}")
    print(f"Token 汇总: {run_dir / 'summaries' / 'variant_token_summary.csv'}")


if __name__ == "__main__":
    main()
