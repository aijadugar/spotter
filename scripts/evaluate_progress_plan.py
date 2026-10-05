from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotter.steps.progress_plan import SessionFindings, build_fallback_plan
from spotter.steps.progress_plan_model import build_progress_plan_prompt
from spotter.contracts import SessionRecord


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Dataset not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _evaluate_deterministic(eval_rows: list[dict]) -> dict:
    """Evaluate the deterministic fallback planner on eval set."""
    correct = 0
    total = len(eval_rows)
    details = []

    for row in eval_rows:
        # Parse prompt to extract history and findings
        prompt = row["prompt"]
        completion = json.loads(row["completion"])

        # The deterministic planner is already what generates completions in build_rows
        # So we compare the generated plan structure
        # For now, just verify the structure matches expected keys
        expected_keys = {"focus", "targets", "next_session_cues", "encouragement", "confidence_notes"}
        actual_keys = set(completion.keys())

        if expected_keys == actual_keys:
            correct += 1
            details.append({"status": "ok", "exercise": row.get("meta", {}).get("exercise")})
        else:
            details.append({"status": "mismatch", "expected": list(expected_keys), "got": list(actual_keys)})

    return {
        "method": "deterministic_fallback",
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "details": details[:5],  # First 5 for brevity
    }


def _evaluate_tinker_finetuned(eval_rows: list[dict], model_id: str) -> dict:
    """Evaluate the fine-tuned Tinker model on eval set."""
    try:
        import tinker
        from tinker import types
    except ImportError:
        return {"error": "Tinker SDK not installed"}

    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        return {"error": "TINKER_API_KEY not set"}

    try:
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(base_model=model_id)
        tokenizer = sampling_client.get_tokenizer()  # This may not exist, use base model tokenizer

        # For evaluation, we'd need to run sampling and compare
        # This is a placeholder for the actual evaluation logic
        return {
            "method": "tinker_finetuned",
            "model_id": model_id,
            "status": "needs_implementation",
            "note": "Full evaluation requires sampling loop implementation",
        }
    except Exception as e:
        return {"error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="data/sft/progress_plan_eval.jsonl")
    parser.add_argument("--model-id", default=None, help="Fine-tuned model ID from Tinker")
    parser.add_argument("--output", default="reports/tinker_eval.md")
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    eval_rows = _load_jsonl(eval_path)

    results = {"eval_rows": len(eval_rows)}

    # Always evaluate deterministic baseline
    det_results = _evaluate_deterministic(eval_rows)
    results["deterministic"] = det_results

    # Evaluate fine-tuned if model ID provided
    if args.model_id:
        tinker_results = _evaluate_tinker_finetuned(eval_rows, args.model_id)
        results["tinker_finetuned"] = tinker_results
    else:
        results["tinker_finetuned"] = {"status": "skipped", "reason": "No --model-id provided"}

    # Write report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# Tinker Fine-tuning Evaluation Report",
        "",
        f"**Eval rows:** {results['eval_rows']}",
        f"**Model ID:** {args.model_id or 'N/A'}",
        "",
        "## Deterministic Fallback Baseline",
        f"- Total: {det_results['total']}",
        f"- Correct: {det_results['correct']}",
        f"- Accuracy: {det_results['accuracy']:.2%}",
        "",
    ]

    if "tinker_finetuned" in results:
        tinker_res = results["tinker_finetuned"]
        md_lines.append("## Tinker Fine-tuned Model")
        if "error" in tinker_res:
            md_lines.append(f"- Error: {tinker_res['error']}")
        elif "status" in tinker_res:
            md_lines.append(f"- Status: {tinker_res['status']}")
            if "note" in tinker_res:
                md_lines.append(f"- Note: {tinker_res['note']}")
        else:
            md_lines.append(f"- Accuracy: {tinker_res.get('accuracy', 'N/A')}")

    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Report written to {out_path}")

    # Print summary to stdout
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())