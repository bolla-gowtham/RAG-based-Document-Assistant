#!/usr/bin/env python
"""CLI: run the evaluation harness against a labeled QA set.

Usage:
    python scripts/evaluate.py --qa-file data/eval_qa.json
    python scripts/evaluate.py --qa-file data/eval_qa.json --no-judge   # retrieval-only, no API calls
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import run_evaluation  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--qa-file", required=True, help="Path to labeled QA JSON file")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge scoring (retrieval metrics only)")
    parser.add_argument("--output", default=None, help="Optional path to save full results as JSON")
    args = parser.parse_args()

    report = run_evaluation(args.qa_file, judge=not args.no_judge)

    print("\n=== Evaluation Summary ===")
    for k, v in report["summary"].items():
        print(f"{k}: {v}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
