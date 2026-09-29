"""
scripts/eval_imt.py
====================
Evaluate IMT audit results (produced by scripts/run_imt.py): AUROC + 5-fold
binary metrics, against either the main DeepSeek-R1 ground truth or the
human-eval target-model ground truth.

Usage:
    # Main dataset — default model (openai_gpt4o)
    python scripts/eval_imt.py

    # Main dataset — specific models
    python scripts/eval_imt.py --models openai_gpt4o anthropic_claude_sonnet46

    # Main dataset — ALL models found in results/imt_audit/
    python scripts/eval_imt.py --all

    # Human-eval datasets — all targets, all auditors found
    python scripts/eval_imt.py --dataset human_eval

    # Human-eval datasets — restrict to one target / specific auditors
    python scripts/eval_imt.py --dataset human_eval --targets gpt4o
    python scripts/eval_imt.py --dataset human_eval --auditors openai_gpt4o anthropic_claude_sonnet46

Output:
    Main dataset:       results/eval/imt/<model>.json, results/eval/imt/summary.txt
    Human-eval dataset: results/eval/imt_human_eval/<target>/<auditor>.json
                        results/eval/imt_human_eval/<target>/summary.txt
                        results/eval/imt_human_eval/summary.txt   (combined across targets)
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import (
    HUMAN_EVAL_AUDIT_RESULTS_DIR,
    HUMAN_EVAL_EVAL_RESULTS_DIR,
    HUMAN_EVAL_TARGETS,
    IMT_AUDIT_RESULTS_DIR,
    EVAL_RESULTS_DIR,
    model_dataset_path,
)
from workflows.imt_scores import (
    CONDITIONS,
    DEFAULT_LABELS,
    RESPONSE_STRATEGY, SIDES, THOUGHT_STRATEGY,
    aggregate_auroc_response_gt_runs,
    aggregate_binary_response_gt_runs,
    compute_auroc_response_gt,
    compute_binary_cv_response_gt,
    compute_oof_predictions,
    find_model_files, load_records,
)

DEFAULT_MODEL = "openai_gpt4o"
OUT_DIR = EVAL_RESULTS_DIR / "imt"
LABELED_OUT_DIR = ROOT / "results" / "imt_audit_with_label"
ALL_TARGETS = HUMAN_EVAL_TARGETS

METRIC_LABELS = {
    "f1":           "Macro-F1",
    "auprc_oof":    "AUPRC (OOF)",
    "acc":          "Accuracy",
    "balanced_acc": "Balanced Accuracy",
    "precision":    "Precision",
    "recall":       "Recall",
    "fpr":          "False Positive Rate",
    "fnr":          "False Negative Rate",
}


# ── Labeled JSON output ───────────────────────────────────────────────────────

def _write_labeled_json(
    result_path: Path,
    records: list,
    predictions: dict,
    out_dir: Path,
) -> Path:
    """Augment an IMT audit result file with OOF eval labels and save."""
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        topic    = item.get("topic", "")
        question = item.get("question", "")
        for l2_type, content in item.get("results", {}).items():
            pred = predictions.get((topic, question, l2_type))
            if pred is None:
                continue
            content["eval"] = {
                "thought":  "decept" if pred["thought"]  == 1 else "honest",
                "response": "decept" if pred["response"] == 1 else "honest",
            }

    out_path = out_dir / result_path.name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out_path


# ── Evaluation ────────────────────────────────────────────────────────────────

def _count_valid_records(records: list) -> dict:
    """Count valid (non-NaN) records per side for metrics calculation."""
    import numpy as np
    counts = {"thought": 0, "response": 0}
    for r in records:
        thought_score = r["score_thought"].get(THOUGHT_STRATEGY, float("nan"))
        response_score = r["score_response"].get(RESPONSE_STRATEGY, float("nan"))
        if not np.isnan(thought_score):
            counts["thought"] += 1
        if not np.isnan(response_score):
            counts["response"] += 1
    return counts


def eval_model(
    model: str,
    files: list,
    labels_path: Path = DEFAULT_LABELS,
    labeled_out_dir: Path = LABELED_OUT_DIR,
) -> dict:
    """
    Run AUROC + 5-fold binary CV across all run files for a model.

    Reports exactly two conditions (see workflows.imt_scores.CONDITIONS),
    both scored against the response's own human label:
      - "response": response-only score (RESPONSE_STRATEGY = full_avg)
      - "combined": "Thought & Response" -- post-hoc mean of the
        independently-computed thought and response scores. No extra LLM
        call or joint-audit logic is involved.
    There is no standalone Thought-vs-its-own-label AUROC.
    """
    print(f"\n  {model}  ({len(files)} run(s))")
    auroc_runs, binary_runs, data_counts = [], [], []

    for path in files:
        records = load_records(path, labels_path=labels_path)
        if not records:
            print(f"    [WARN] No valid records in {path.name}")
            continue
        valid_counts = _count_valid_records(records)
        print(f"    {path.name}: {len(records)} records  [thought={valid_counts['thought']}, response={valid_counts['response']} valid]")
        auroc_runs.append(compute_auroc_response_gt(records))
        binary_runs.append(compute_binary_cv_response_gt(records))
        data_counts.append({"file": path.name, "total": len(records), **valid_counts})

        # Generate labeled JSON with OOF predictions
        predictions = compute_oof_predictions(records)
        labeled_path = _write_labeled_json(path, records, predictions, labeled_out_dir)
        print(f"    labeled -> {labeled_path.relative_to(ROOT)}")

    if not auroc_runs:
        return {}

    if len(auroc_runs) == 1:
        auroc_result  = {"single": auroc_runs[0]}
        binary_result = {"single": binary_runs[0]}
    else:
        auroc_result  = aggregate_auroc_response_gt_runs(auroc_runs)
        binary_result = aggregate_binary_response_gt_runs(binary_runs)

    return {
        "auroc": auroc_result, "binary": binary_result,
        "runs": len(auroc_runs), "data_counts": data_counts,
    }


# ── Summary formatting ────────────────────────────────────────────────────────

def _get(block: dict, key: str) -> tuple[float, float]:
    """Return (mean, std) from a metric block."""
    if "mean" in block:
        return block["mean"].get(key, float("nan")), block["std"].get(key, float("nan"))
    if "single" in block:
        m = block["single"].get(f"{key}_mean", block["single"].get(key, float("nan")))
        s = block["single"].get(f"{key}_std", float("nan"))
        return m, s
    return float("nan"), float("nan")


def _get_response_gt_auroc(auroc_block: dict, condition: str, domain: str = "Overall") -> tuple[float, float]:
    if "mean" in auroc_block:
        return auroc_block["mean"][condition][domain], auroc_block["std"][condition][domain]
    if "single" in auroc_block:
        return auroc_block["single"][condition][domain], float("nan")
    return float("nan"), float("nan")


def _fmt(mean: float, std: float) -> str:
    if mean != mean:   # nan check
        return "   NA   "
    return f"{mean:.4f}±{std:.4f}"


CONDITION_LABELS = {"response": "Response", "combined": "Thought & Response"}


def build_summary(all_results: dict) -> str:
    lines = []
    lines.append("IMT Evaluation Summary")
    lines.append(f"Config: response={RESPONSE_STRATEGY}, thought={THOUGHT_STRATEGY} (used only inside Thought & Response), 5-fold CV, domain-specific threshold")
    lines.append("Ground truth: human_eval.response for both conditions below (Thought & Response = mean(thought, response) score)")
    lines.append("")

    for model, result in sorted(all_results.items()):
        if not result:
            continue
        lines.append(f"Model: {model}  ({result['runs']} run(s))")

        # Data availability summary
        if "data_counts" in result:
            for dc in result["data_counts"]:
                lines.append(f"  {dc['file']}: {dc['total']} total, thought={dc['thought']} valid, response={dc['response']} valid")

        auroc_b  = result["auroc"]
        binary_b = result["binary"]

        for condition in CONDITIONS:
            auroc_m, auroc_s = _get_response_gt_auroc(auroc_b, condition)

            lines.append(f"  {CONDITION_LABELS[condition]}")
            lines.append(f"    AUROC         : {_fmt(auroc_m, auroc_s)}")

            for key, label in METRIC_LABELS.items():
                if "mean" in binary_b:
                    m = binary_b["mean"][condition].get(key, float("nan"))
                    s = binary_b["std"][condition].get(key, float("nan"))
                else:
                    base = binary_b["single"][condition]
                    if key == "auprc_oof":
                        m = base.get("auprc_oof", float("nan"))
                        s = float("nan")
                    else:
                        m = base.get(f"{key}_mean", float("nan"))
                        s = base.get(f"{key}_std",  float("nan"))
                lines.append(f"    {label:<20}: {_fmt(m, s)}")

        lines.append("")

    return "\n".join(lines)


# ── Dataset branches ─────────────────────────────────────────────────────────

def _eval_main_dataset(args) -> None:
    model_filter = None if args.all else args.models
    model_files  = find_model_files(IMT_AUDIT_RESULTS_DIR, model_filter)

    if not model_files:
        print(f"[ERROR] No IMT result files found in {IMT_AUDIT_RESULTS_DIR}")
        sys.exit(1)

    print(f"Found {len(model_files)} model(s): {', '.join(sorted(model_files))}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LABELED_OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model, files in sorted(model_files.items()):
        result = eval_model(model, files)
        all_results[model] = result
        out = OUT_DIR / f"{model}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"    -> {out.relative_to(ROOT)}")

    summary_path = OUT_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(build_summary(all_results))
    print(f"\nSummary: {summary_path.relative_to(ROOT)}")


def _eval_human_eval_dataset(args) -> None:
    HUMAN_EVAL_EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined_lines = []

    for target in args.targets:
        results_dir = HUMAN_EVAL_AUDIT_RESULTS_DIR / target
        labels_path = model_dataset_path(target)

        if not results_dir.exists():
            print(f"[WARN] No audit results dir for target '{target}': {results_dir}")
            continue

        model_files = find_model_files(results_dir, args.auditors)
        if not model_files:
            print(f"[WARN] No IMT result files found in {results_dir}")
            continue

        print(f"\n=== target: {target}  ({len(model_files)} auditor(s)) ===")
        out_dir = HUMAN_EVAL_EVAL_RESULTS_DIR / target
        labeled_out_dir = ROOT / "results" / "imt_audit_human_eval_with_label" / target
        out_dir.mkdir(parents=True, exist_ok=True)
        labeled_out_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}
        for auditor, files in sorted(model_files.items()):
            result = eval_model(auditor, files, labels_path=labels_path, labeled_out_dir=labeled_out_dir)
            all_results[auditor] = result
            if not result:
                continue

            out = out_dir / f"{auditor}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"    -> {out.relative_to(ROOT)}")

        # Merge in previously cached per-auditor results (e.g. from an earlier run with
        # a different --auditors filter) so the summary always covers everything on disk.
        for cached in out_dir.glob("*.json"):
            auditor = cached.stem
            if auditor not in all_results:
                with open(cached, encoding="utf-8") as f:
                    all_results[auditor] = json.load(f)

        summary = build_summary(all_results)
        summary_path = out_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"  Summary: {summary_path.relative_to(ROOT)}")

        combined_lines.append(f"##### target: {target} #####\n")
        combined_lines.append(summary)
        combined_lines.append("\n")

    combined_path = HUMAN_EVAL_EVAL_RESULTS_DIR / "summary.txt"
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_lines))
    print(f"\nCombined summary: {combined_path.relative_to(ROOT)}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IMT audit results")
    parser.add_argument("--dataset", choices=("main", "human_eval"), default="main",
                        help="Which dataset to evaluate against (default: main)")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL],
                        help=f"[--dataset main] Model preset keys to evaluate (default: {DEFAULT_MODEL})")
    parser.add_argument("--all", action="store_true",
                        help="[--dataset main] Evaluate all models found in results/imt_audit/")
    parser.add_argument("--targets", nargs="+", default=ALL_TARGETS, choices=ALL_TARGETS,
                        help=f"[--dataset human_eval] Target-model datasets to evaluate (default: all of {ALL_TARGETS})")
    parser.add_argument("--auditors", nargs="+", default=None,
                        help="[--dataset human_eval] Auditor preset keys to evaluate (default: all found on disk)")
    args = parser.parse_args()

    if args.dataset == "human_eval":
        _eval_human_eval_dataset(args)
    else:
        _eval_main_dataset(args)


if __name__ == "__main__":
    main()
