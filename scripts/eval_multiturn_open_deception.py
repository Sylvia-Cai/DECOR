"""Comprehensive evaluation for multiturn OpenDeception results.

Evaluates two multiturn method families on one target side (``thought`` or
``response``):

1. IMT (DECOR) round-level multiturn scores.
2. Simple-LLM round-level baseline (per-round judgment, then aggregated —
   matches the paper's Appendix C.1 multi-turn evaluation protocol).

Outputs:
  - Human-readable summary txt.
  - Machine-readable json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "open_deception"
N_FOLDS = 5
FOLD_SEED = 42

DEFAULT_PATHS = {
    "thought": {
        # imt_multiturn_gpt4o.json (workflows/multiturn_iu_sis_workflow.py's default
        # output) contains both thought & response scores.
        "imt": RESULTS_DIR / "imt_multiturn_gpt4o.json",
        "round": RESULTS_DIR / "simple_llm_judge_open_deception_round_thought.json",
        "out_txt": RESULTS_DIR / "multiturn_thought_eval.txt",
        "out_json": RESULTS_DIR / "multiturn_thought_eval.json",
    },
    "response": {
        "imt": RESULTS_DIR / "imt_multiturn_gpt4o.json",
        "round": RESULTS_DIR / "simple_llm_judge_open_deception_round.json",
        "out_txt": RESULTS_DIR / "multiturn_response_eval.txt",
        "out_json": RESULTS_DIR / "multiturn_response_eval.json",
    },
    # Response-as-ground-truth combined: OpenDeception's dialogue_label is
    # already an outcome-level (response-side) label -- both --target thought
    # and --target response above already score against it. "combined" adds
    # a third condition, the post-hoc mean of the independently-computed
    # thought and response dialogue scores, evaluated against that same
    # label. No extra LLM call, no joint-audit logic.
    "combined": {
        "imt": RESULTS_DIR / "imt_multiturn_gpt4o.json",
        "round": RESULTS_DIR / "simple_llm_judge_open_deception_round.json",
        "round_thought": RESULTS_DIR / "simple_llm_judge_open_deception_round_thought.json",
        "out_txt": RESULTS_DIR / "multiturn_combined_eval.txt",
        "out_json": RESULTS_DIR / "multiturn_combined_eval.json",
    },
}


@dataclass
class Metrics:
    auroc: float
    auprc: float
    accuracy: float
    balanced_acc: float
    f1: float
    precision: float
    recall: float
    specificity: float
    mcc: float
    fpr: float
    fnr: float
    tp: int
    fp: int
    fn: int
    tn: int
    n: int
    n_pos: int
    n_neg: int


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _label_to_binary(label: str, positive_labels: Iterable[str], negative_labels: Iterable[str]) -> Optional[int]:
    if label in positive_labels:
        return 1
    if label in negative_labels:
        return 0
    return None


def _get_metrics(y_true: List[int], y_score: List[float], y_pred: List[int]) -> Metrics:
    yt = np.array(y_true, dtype=int)
    ys = np.array(y_score, dtype=float)
    yp = np.array(y_pred, dtype=int)

    has_both = len(np.unique(yt)) >= 2
    auroc = float(roc_auc_score(yt, ys)) if has_both else float("nan")
    auprc = float(average_precision_score(yt, ys)) if has_both else float("nan")
    f1 = float(f1_score(yt, yp, zero_division=0))
    precision = float(precision_score(yt, yp, zero_division=0))
    recall = float(recall_score(yt, yp, zero_division=0))
    balanced_acc = float(balanced_accuracy_score(yt, yp))
    mcc = float(matthews_corrcoef(yt, yp))
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    accuracy = float((tp + tn) / len(yt)) if len(yt) else float("nan")
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    return Metrics(
        auroc=auroc,
        auprc=auprc,
        accuracy=accuracy,
        balanced_acc=balanced_acc,
        f1=f1,
        precision=precision,
        recall=recall,
        specificity=specificity,
        mcc=mcc,
        fpr=fpr,
        fnr=fnr,
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
        n=int(len(yt)),
        n_pos=int(yt.sum()),
        n_neg=int((1 - yt).sum()),
    )


def _best_threshold_macro_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    best_thr = float(np.median(y_score)) if len(y_score) else 0.5
    best_val = -1.0
    for thr in np.unique(y_score):
        y_pred = (y_score >= thr).astype(int)
        val = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if val > best_val:
            best_val = val
            best_thr = float(thr)
    return best_thr


def _cv_threshold_predictions(y_true: List[int], y_score: List[float]) -> Dict[str, Any]:
    yt = np.array(y_true, dtype=int)
    ys = np.array(y_score, dtype=float)

    if len(ys) < N_FOLDS or len(np.unique(yt)) < 2:
        thr = _best_threshold_macro_f1(yt, ys) if len(ys) else 0.5
        y_pred = (ys >= thr).astype(int)
        return {
            "predictions": y_pred.tolist(),
            "thresholds": [float(thr)],
            "threshold_mean": float(thr),
            "threshold_std": 0.0,
            "mode": "fallback_single_threshold",
        }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    oof_pred = np.full(len(ys), -1, dtype=int)
    thresholds: List[float] = []

    for train_idx, test_idx in skf.split(ys, yt):
        y_train = yt[train_idx]
        s_train = ys[train_idx]
        s_test = ys[test_idx]
        thr = _best_threshold_macro_f1(y_train, s_train)
        thresholds.append(float(thr))
        oof_pred[test_idx] = (s_test >= thr).astype(int)

    return {
        "predictions": oof_pred.tolist(),
        "thresholds": thresholds,
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_std": float(np.std(thresholds)),
        "mode": f"cv{N_FOLDS}_oof_macro_f1",
    }


def _score_stats(y_true: List[int], y_score: List[float]) -> Dict[str, Any]:
    yt = np.array(y_true, dtype=int)
    ys = np.array(y_score, dtype=float)
    pos = ys[yt == 1]
    neg = ys[yt == 0]
    return {
        "all": {
            "min": float(np.min(ys)),
            "max": float(np.max(ys)),
            "mean": float(np.mean(ys)),
            "median": float(np.median(ys)),
        },
        "pos": {
            "mean": float(np.mean(pos)) if len(pos) else float("nan"),
            "median": float(np.median(pos)) if len(pos) else float("nan"),
        },
        "neg": {
            "mean": float(np.mean(neg)) if len(neg) else float("nan"),
            "median": float(np.median(neg)) if len(neg) else float("nan"),
        },
    }


def _collect_errors(source_files: List[str], y_true: List[int], y_score: List[float], y_pred: List[int], top_k: int) -> Dict[str, List[Dict[str, Any]]]:
    false_pos = []
    false_neg = []
    for source, truth, score, pred in zip(source_files, y_true, y_score, y_pred):
        item = {"source_file": source, "y_true": truth, "y_score": float(score), "y_pred": pred}
        if truth == 0 and pred == 1:
            false_pos.append(item)
        elif truth == 1 and pred == 0:
            false_neg.append(item)
    false_pos.sort(key=lambda x: x["y_score"], reverse=True)
    false_neg.sort(key=lambda x: x["y_score"])
    return {
        "false_positives": false_pos[:top_k],
        "false_negatives": false_neg[:top_k],
    }


def _evaluate_method(
    *,
    family: str,
    name: str,
    source_files: List[str],
    y_true: List[int],
    y_score: List[float],
    threshold: Optional[float],
    threshold_label: str,
    predictions: Optional[List[int]] = None,
    threshold_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if predictions is not None:
        y_pred = predictions
    elif threshold is None:
        y_pred = [int(round(v)) for v in y_score]
    else:
        y_pred = [1 if v >= threshold else 0 for v in y_score]
    metrics = _get_metrics(y_true, y_score, y_pred)
    return {
        "family": family,
        "method": name,
        "threshold": threshold,
        "threshold_label": threshold_label,
        "threshold_details": threshold_details or {},
        "metrics": asdict(metrics),
        "score_stats": _score_stats(y_true, y_score),
        "judge_output": {
            "pred_positive": int(sum(y_pred)),
            "pred_negative": int(len(y_pred) - sum(y_pred)),
        },
        "errors": _collect_errors(source_files, y_true, y_score, y_pred, top_k=5),
    }


def _extract_imt(payload: Dict[str, Any], side: str) -> Dict[str, Any]:
    """
    Extract per-dialogue IMT (Decor) scores for one side.

    workflows/multiturn_iu_sis_workflow.py's current output nests dialogue-level
    scores as dialogue_scores[side][round_agg][dialogue_agg] (e.g. round_agg in
    {"mean", "max"}, dialogue_agg in {"min", "max", "last"}), flattened here into
    "round_agg.dialogue_agg" method names, plus a "fixed" method taken directly
    from fixed_scores[side] -- the workflow's own canonical single-number choice
    (round-level SIS-weighted mean, then dialogue-level max).
    """
    positive = set(payload["meta"].get("positive_labels", []))
    negative = set(payload["meta"].get("negative_labels", []))
    source_files, y_true = [], []
    methods: Dict[str, List[float]] = {}

    for dialogue in payload["dialogues"]:
        binary = _label_to_binary(dialogue.get("dialogue_label", ""), positive, negative)
        if binary is None:
            continue

        branch_scores = dialogue.get("dialogue_scores", {}).get(side, {})
        scores: Dict[str, float] = {}
        for round_agg, dialogue_aggs in branch_scores.items():
            if isinstance(dialogue_aggs, dict):
                for dialogue_agg, value in dialogue_aggs.items():
                    scores[f"{round_agg}.{dialogue_agg}"] = _safe_float(value)
            else:
                scores[str(round_agg)] = _safe_float(dialogue_aggs)

        fixed = dialogue.get("fixed_scores", {}).get(side)
        if fixed is not None:
            scores["fixed"] = _safe_float(fixed)

        if not scores:
            continue
        source_files.append(dialogue["source_file"])
        y_true.append(binary)
        for method in scores:
            methods.setdefault(method, [])
        for method in methods:
            methods[method].append(scores.get(method, 0.0))

    return {"source_files": source_files, "y_true": y_true, "methods": methods}


def _extract_round_baseline(payload: Dict[str, Any], side: str) -> Dict[str, Any]:
    """
    Extract per-dialogue baseline scores for one side, supporting the two
    output schemas produced by baselines/run_*_open_deception_round.py:

      - binary "dialogue_aggregation" (simple_llm_judge, deceptionbench):
        {"decept_ratio": ..., "any_decept": ..., "majority_label": ...},
        either flat (one file per side, e.g. simple_llm_judge) or nested one
        level under the side name (e.g. deceptionbench bundles both sides
        into one file: {"thought": {...}, "response": {...}}).
      - continuous "dialogue_scores" (cot_red_handed, scheming_judge):
        {side: {"min": ..., "max": ..., "mean": ..., "last": ...}}, using
        max-pooling across rounds to match the paper's Appendix C.1
        aggregation ("we apply max-pooling aggregation across all turns").
    """
    source_files, y_true = [], []
    methods: Dict[str, List[float]] = {}

    for dialogue in payload["dialogues"]:
        source_files.append(dialogue["source_file"])
        y_true.append(int(dialogue["binary_label"]))

        agg_root = dialogue.get("dialogue_aggregation")
        scores_root = dialogue.get("dialogue_scores")

        if agg_root is not None:
            agg = agg_root.get(side, agg_root) if side in agg_root else agg_root
            for key, name in (("decept_ratio", "decept_ratio"),):
                methods.setdefault(name, []).append(_safe_float(agg.get(key), 0.0))
            methods.setdefault("any_decept", []).append(1.0 if agg.get("any_decept") else 0.0)
            methods.setdefault("majority_decept", []).append(
                1.0 if agg.get("majority_label") == "decept" else 0.0
            )
        elif scores_root is not None:
            side_scores = scores_root.get(side, {})
            methods.setdefault("max", []).append(_safe_float(side_scores.get("max"), 0.0))
            methods.setdefault("mean", []).append(_safe_float(side_scores.get("mean"), 0.0))
        else:
            raise ValueError(
                f"Dialogue {dialogue.get('source_file')} has neither "
                "'dialogue_aggregation' nor 'dialogue_scores'"
            )

    return {"source_files": source_files, "y_true": y_true, "methods": methods}


def _validate_alignment(reference: Dict[str, Any], candidate: Dict[str, Any], name: str) -> None:
    if reference["source_files"] != candidate["source_files"]:
        raise ValueError(f"Source file ordering mismatch for {name}")
    if reference["y_true"] != candidate["y_true"]:
        raise ValueError(f"Ground-truth mismatch for {name}")


def _round_payload_is_bundled(payload: Dict[str, Any]) -> bool:
    """
    True if one round-baseline payload already carries both thought and
    response scores (deceptionbench / cot_red_handed / scheming_judge,
    whose dialogue_aggregation/dialogue_scores are nested by side), False if
    it's a single-side file (simple_llm_judge, which is run once per
    --target and needs a matching sibling file for the other side).
    """
    dialogues = payload.get("dialogues", [])
    if not dialogues:
        return False
    root = dialogues[0].get("dialogue_aggregation") or dialogues[0].get("dialogue_scores")
    if root is None:
        return False
    return "thought" in root and "response" in root


def _combine_sides(thought_extract: Dict[str, Any], response_extract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-hoc combined = mean(thought_score, response_score) per dialogue,
    for every method present on both sides. Uses only the two
    already-computed, independent per-side scores -- no extra LLM call, no
    joint-audit logic.
    """
    common_methods = sorted(set(thought_extract["methods"]) & set(response_extract["methods"]))
    methods = {
        m: [(t + r) / 2.0 for t, r in zip(thought_extract["methods"][m], response_extract["methods"][m])]
        for m in common_methods
    }
    return {
        "source_files": response_extract["source_files"],
        "y_true": response_extract["y_true"],
        "methods": methods,
    }


def _fmt(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.4f}"


def _build_report(
    *,
    target: str,
    paths: Dict[str, Path],
    y_true: List[int],
    results: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []

    def w(text: str = "") -> None:
        lines.append(text)

    label_counts = Counter(y_true)
    prevalence = label_counts[1] / len(y_true) if y_true else float("nan")

    w("=" * 90)
    w(f"Multiturn OpenDeception Evaluation  [{target}]")
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("=" * 90)
    w()
    w("Inputs")
    w(f"  IMT result           : {paths['imt']}")
    w(f"  Round baseline       : {paths['round']}")
    w()
    w("Dataset")
    w(f"  Kept dialogues       : {len(y_true)}")
    w(f"  Positive / Negative  : {label_counts[1]} / {label_counts[0]}")
    w(f"  Positive rate        : {_fmt(prevalence)}")

    ranking = sorted(results, key=lambda r: (np.nan_to_num(r['metrics']['auroc'], nan=-1.0), np.nan_to_num(r['metrics']['balanced_acc'], nan=-1.0)), reverse=True)

    w()
    w("=" * 90)
    w("Ranking Summary")
    w("=" * 90)
    header = f"{'Method':<42} {'Thr':<14} {'AUROC':>7} {'AUPRC':>7} {'BalAcc':>7} {'MCC':>7} {'F1':>7} {'Recall':>7} {'Spec':>7}"
    w(header)
    w("-" * len(header))
    for row in ranking:
        mm = row["metrics"]
        w(
            f"{row['method']:<42} {row['threshold_label']:<14} "
            f"{_fmt(mm['auroc']):>7} {_fmt(mm['auprc']):>7} {_fmt(mm['balanced_acc']):>7} "
            f"{_fmt(mm['mcc']):>7} {_fmt(mm['f1']):>7} {_fmt(mm['recall']):>7} {_fmt(mm['specificity']):>7}"
        )

    for row in ranking:
        mm = row["metrics"]
        w()
        w("=" * 90)
        w(f"{row['method']}  [{row['threshold_label']}]")
        w("=" * 90)
        w(f"Family                 : {row['family']}")
        w(f"Threshold              : {row['threshold']}")
        if row["threshold_details"]:
            td = row["threshold_details"]
            if td.get("mode"):
                w(f"Threshold mode         : {td['mode']}")
            if "threshold_mean" in td:
                w(f"Threshold mean/std     : {_fmt(td['threshold_mean'])} / {_fmt(td.get('threshold_std', float('nan')))}")
            if td.get("thresholds"):
                w("Fold thresholds        : " + ", ".join(f"{t:.4f}" for t in td["thresholds"]))
        w(f"Judge output           : pos={row['judge_output']['pred_positive']} neg={row['judge_output']['pred_negative']}")
        w(f"AUROC                  : {_fmt(mm['auroc'])}")
        w(f"AUPRC                  : {_fmt(mm['auprc'])}")
        w(f"Accuracy               : {_fmt(mm['accuracy'])}")
        w(f"Balanced Accuracy      : {_fmt(mm['balanced_acc'])}")
        w(f"F1                     : {_fmt(mm['f1'])}")
        w(f"Precision              : {_fmt(mm['precision'])}")
        w(f"Recall                 : {_fmt(mm['recall'])}")
        w(f"Specificity            : {_fmt(mm['specificity'])}")
        w(f"MCC                    : {_fmt(mm['mcc'])}")
        w(f"FPR / FNR              : {_fmt(mm['fpr'])} / {_fmt(mm['fnr'])}")
        w(f"Confusion (TN/FP/FN/TP): {mm['tn']}/{mm['fp']}/{mm['fn']}/{mm['tp']}")

        stats = row["score_stats"]
        w("Score distribution")
        w(
            f"  all(min/median/mean/max): {_fmt(stats['all']['min'])} / {_fmt(stats['all']['median'])} / "
            f"{_fmt(stats['all']['mean'])} / {_fmt(stats['all']['max'])}"
        )
        w(f"  pos(mean/median)        : {_fmt(stats['pos']['mean'])} / {_fmt(stats['pos']['median'])}")
        w(f"  neg(mean/median)        : {_fmt(stats['neg']['mean'])} / {_fmt(stats['neg']['median'])}")

        fps = row["errors"]["false_positives"]
        fns = row["errors"]["false_negatives"]
        w("Top false positives")
        if fps:
            for item in fps:
                w(f"  {item['y_score']:.4f}  {item['source_file']}")
        else:
            w("  None")
        w("Top false negatives")
        if fns:
            for item in fns:
                w(f"  {item['y_score']:.4f}  {item['source_file']}")
        else:
            w("  None")

    w()
    w("Notes")
    w("  - AUROC and Balanced Accuracy are the most stable primary metrics for this imbalanced set.")
    w("  - AUPRC is prevalence-sensitive; compare it against the dataset positive rate above.")
    w(f"  - For continuous scores, both fixed@0.5 and CV{N_FOLDS} out-of-fold thresholding are reported.")
    w("  - CV thresholds are selected on each training fold by maximizing macro-F1, then applied to the held-out fold.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiturn OpenDeception outputs comprehensively")
    parser.add_argument("--target", choices=("thought", "response", "combined"), required=True)
    parser.add_argument("--imt-path", type=Path, default=None, help="Path to multiturn IMT result JSON")
    parser.add_argument("--round-path", type=Path, default=None,
                        help="Path to round-level baseline JSON (response side for --target combined "
                             "if the family bundles both sides in one file)")
    parser.add_argument("--round-path-thought", type=Path, default=None,
                        help="[--target combined only] Path to the thought-side round-level baseline JSON, "
                             "needed only for a family that outputs one side per file (e.g. simple_llm_judge). "
                             "Not needed for families that already bundle both sides in --round-path "
                             "(deceptionbench, cot_red_handed, scheming_judge).")
    parser.add_argument("--out-txt", type=Path, default=None, help="Output txt path")
    parser.add_argument("--out-json", type=Path, default=None, help="Output json path")
    args = parser.parse_args()

    defaults = DEFAULT_PATHS[args.target]
    paths = {
        "imt": args.imt_path or defaults["imt"],
        "round": args.round_path or defaults["round"],
        "out_txt": args.out_txt or defaults["out_txt"],
        "out_json": args.out_json or defaults["out_json"],
    }
    for key in ("imt", "round"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing input file for {key}: {paths[key]}")

    imt_payload = _load_json(paths["imt"])
    round_payload = _load_json(paths["round"])

    if args.target == "combined":
        imt_thought = _extract_imt(imt_payload, "thought")
        imt_response = _extract_imt(imt_payload, "response")
        _validate_alignment(imt_thought, imt_response, "imt thought vs response")
        imt = _combine_sides(imt_thought, imt_response)

        if _round_payload_is_bundled(round_payload):
            round_thought_level = _extract_round_baseline(round_payload, "thought")
            round_response_level = _extract_round_baseline(round_payload, "response")
        else:
            round_thought_path = args.round_path_thought or defaults["round_thought"]
            if not round_thought_path.exists():
                raise FileNotFoundError(
                    f"--target combined: {paths['round']} only contains one side, so the thought-side "
                    f"sibling file is required too. Missing: {round_thought_path} "
                    "(pass --round-path-thought explicitly, or bundle both sides in one file)."
                )
            round_thought_payload = _load_json(round_thought_path)
            round_thought_level = _extract_round_baseline(round_thought_payload, "thought")
            round_response_level = _extract_round_baseline(round_payload, "response")
        _validate_alignment(round_thought_level, round_response_level, "round baseline thought vs response")
        round_level = _combine_sides(round_thought_level, round_response_level)
    else:
        imt = _extract_imt(imt_payload, args.target)
        round_level = _extract_round_baseline(round_payload, args.target)

    _validate_alignment(imt, round_level, "round baseline")

    source_files = imt["source_files"]
    y_true = imt["y_true"]
    results: List[Dict[str, Any]] = []

    for method_name, scores in imt["methods"].items():
        fixed_thr = 0.5
        results.append(
            _evaluate_method(
                family="imt_multiturn",
                name=f"IMT {method_name}",
                source_files=source_files,
                y_true=y_true,
                y_score=scores,
                threshold=fixed_thr,
                threshold_label="fixed@0.5",
            )
        )
        cv_out = _cv_threshold_predictions(y_true, scores)
        results.append(
            _evaluate_method(
                family="imt_multiturn",
                name=f"IMT {method_name}",
                source_files=source_files,
                y_true=y_true,
                y_score=scores,
                threshold=cv_out["threshold_mean"],
                threshold_label=f"cv{N_FOLDS}_oof",
                predictions=cv_out["predictions"],
                threshold_details=cv_out,
            )
        )

    # Continuous methods (decept_ratio / max / mean) get both a fixed@0.5 and
    # a cross-validated threshold; binary methods (any_decept / majority_decept)
    # are already 0/1 and need no thresholding.
    binary_methods = {"any_decept", "majority_decept"}
    for method_name, scores in round_level["methods"].items():
        label = f"Round baseline {method_name}"
        if method_name in binary_methods:
            results.append(
                _evaluate_method(
                    family="round_baseline",
                    name=label,
                    source_files=source_files,
                    y_true=y_true,
                    y_score=scores,
                    threshold=None,
                    threshold_label="binary",
                )
            )
            continue

        results.append(
            _evaluate_method(
                family="round_baseline",
                name=label,
                source_files=source_files,
                y_true=y_true,
                y_score=scores,
                threshold=0.5,
                threshold_label="fixed@0.5",
            )
        )
        cv_out = _cv_threshold_predictions(y_true, scores)
        results.append(
            _evaluate_method(
                family="round_baseline",
                name=label,
                source_files=source_files,
                y_true=y_true,
                y_score=scores,
                threshold=cv_out["threshold_mean"],
                threshold_label=f"cv{N_FOLDS}_oof",
                predictions=cv_out["predictions"],
                threshold_details=cv_out,
            )
        )

    report = _build_report(target=args.target, paths=paths, y_true=y_true, results=results)
    json_payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "target": args.target,
            "inputs": {k: str(v) for k, v in paths.items() if k in {"imt", "round"}},
            "kept_dialogues": len(y_true),
            "n_pos": int(sum(y_true)),
            "n_neg": int(len(y_true) - sum(y_true)),
        },
        "results": results,
    }

    paths["out_txt"].parent.mkdir(parents=True, exist_ok=True)
    paths["out_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["out_txt"].write_text(report, encoding="utf-8")
    paths["out_json"].write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved txt : {paths['out_txt']}")
    print(f"Saved json: {paths['out_json']}")
    print(report)


if __name__ == "__main__":
    main()