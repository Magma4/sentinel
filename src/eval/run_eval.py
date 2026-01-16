import json
import os
import time
from typing import List, Dict, Any
from src.core.schema import PatientRecord, GroundTruthItem, SafetyFlag
from src.core.audit import SafetyAuditor
from src.core.extract import FactExtractor
from src.eval.metrics import precision_recall_f1, severity_weighted_recall, evidence_grounding_rate

RESULTS_FILE = "results.json"
SUMMARY_FILE = "results.md"

def match_flag_to_ground_truth(flag: SafetyFlag, ground_truth: List[GroundTruthItem]) -> GroundTruthItem | None:
    """
    Matches a detected flag to a ground truth item.
    Rules:
    1. Category must match.
    2. Explanation must contain the Ground Truth 'key' (case-insensitive substring).
    """
    for gt in ground_truth:
        if flag.category.value != gt.category:
            continue

        # Semantic check: Is the specific medical concept (key) present in the finding?
        # e.g. key="Metformin" in explanation "Contraindicated Metformin use..."
        if gt.key.lower() in flag.explanation.lower():
            return gt

    return None

def generate_markdown_report(results: Dict):
    """Generates a markdown summary of the evaluation results."""
    summary = results["summary"]
    md = "# SentinelMD Evals\n\n"
    md += f"**Cases**: {summary['total_cases']} | **Avg Runtime**: {summary['avg_runtime_sec']}s\n\n"
    md += "## Aggregate Metrics\n"
    md += f"- **F1 Score**: {summary['f1']}\n"
    md += f"- **Precision**: {summary['precision']}\n"
    md += f"- **Recall**: {summary['recall']}\n"
    md += f"- **Weighted Recall**: {summary['weighted_recall']}\n"
    md += f"- **High-Severity Recall**: {summary['high_severity_recall']}\n"
    md += f"- **False Positive Rate (FDR)**: {summary['avg_fpr_fdr']}\n"
    md += f"- **Evidence Grounding**: {summary['grounding_rate']}\n\n"
    md += "## Case Breakdown\n"
    md += "| Case | TP | FP | FN | F1 | W.Rec | H.Rec | FDR | Ground |\n"
    md += "|---|---|---|---|---|---|---|---|---|\n"
    for c in results["cases"]:
        m = c["metrics"]
        md += f"| {c['filename']} | {c['tp']} | {c['fp']} | {c['fn']} | {m['f1']} | {m['weighted_recall']} | {m['high_severity_recall']} | {m['fdr']} | {m['grounding_rate']} |\n"

    with open(SUMMARY_FILE, "w") as f:
        f.write(md)

def run_eval_pipeline():
    """
    Runs the full extraction -> audit pipeline against all synthetic cases.
    """
    data_dir = "data/synthetic"

    # Initialize Pipeline Components (Mock Mode by default)
    # To test LLM, one would pass backend_url here, but let's default to Mock for automated CI/eval.
    extractor = FactExtractor()
    auditor = SafetyAuditor()

    print("Starting Evaluation Pipeline...")

    results = {
        "summary": {},
        "cases": []
    }

    # Aggregators
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_weighted_recall_sum = 0.0
    total_grounding_rate_sum = 0.0
    total_runtime = 0.0
    valid_cases_count = 0

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r") as f:
            data = json.load(f)

        record = PatientRecord(**data)
        ground_truth = record.ground_truth

        # 1. Prepare Raw Inputs
        note_text = "\n".join([n.content for n in record.notes])
        labs_text = "\n".join([f"{l.name}: {l.value} {l.unit}" for l in record.labs])
        meds_text = ", ".join(record.medications)

        # 2. Execute Pipeline & Measure Time
        start_time = time.time()

        # Step A: Extract
        extracted_facts = extractor.extract_facts(note_text, labs_text, meds_text)

        # Step B: Audit
        report = auditor.run_audit(extracted_facts, note_text, labs_text, meds_text)

        duration = time.time() - start_time
        total_runtime += duration

        # 3. Calculate Case Metrics
        tp_count = 0
        fp_count = 0

        matched_gt_items = []
        matched_flag_indices = set()

        # Check TPs and FPs
        for i, flag in enumerate(report.flags):
            match = match_flag_to_ground_truth(flag, ground_truth)
            if match:
                tp_count += 1
                matched_gt_items.append(match)
                matched_flag_indices.add(i)
            else:
                fp_count += 1

        # Check FNs (Ground Truths not matched)
        # Note: A single flag might match multiple GTs if ambiguous, but here we scan flags.
        # Better: Scan GTs to see coverage?
        # Current logic: One flag = one finding. If one flag covers a GT, that GT is found.
        # If we have duplicate flags for same GT, simplest is count unique matched GTs.
        unique_matched_gt_ids = set([id(m) for m in matched_gt_items]) # simplistic unique check
        fn_count = len(ground_truth) - len(unique_matched_gt_ids)

        # Re-calc TP based on unique discovered risk (to avoid gaming by duplicate flags)
        # Actually, let's keep it simple: TP = number of flags that are valid hits.
        # But for F1, we usually want Recall vs GT.
        # Recall = (Unique GT Found) / (Total GT)
        # Precision = (Valid Flags) / (Total Flags)

        # Let's align counts strictly
        unique_matched_gts = []
        seen_gt_ids = set()
        for m in matched_gt_items:
            if id(m) not in seen_gt_ids:
                seen_gt_ids.add(id(m))
                unique_matched_gts.append(m)

        case_tp = len(unique_matched_gts)
        case_fp = fp_count # Flags that matched nothing
        case_fn = len(ground_truth) - case_tp

        precision, recall, f1 = precision_recall_f1(case_tp, case_fp, case_fn)
        w_recall = severity_weighted_recall(unique_matched_gts, ground_truth)

        # New Metrics
        from src.eval.metrics import high_severity_recall
        h_recall = high_severity_recall(unique_matched_gts, ground_truth)

        # False Positive Rate (per case) => Here approximated as False Discovery Rate (FP / (TP+FP))?
        # Or just raw FP count? User asked for "False Positive Rate per case".
        # Let's provide "False Discovery Rate" (1 - Precision) as a rate [0-1].
        # If no flags, FDR is 0.
        denom = (case_tp + case_fp)
        fdr = case_fp / denom if denom > 0 else 0.0

        # Grounding Rate
        grounded_flags_count = 0
        for flag in report.flags:
            if evidence_grounding_rate(flag, note_text, labs_text, meds_text):
                grounded_flags_count += 1
        case_grounding = grounded_flags_count / len(report.flags) if report.flags else 1.0

        # Accumulate
        total_tp += case_tp
        total_fp += case_fp
        total_fn += case_fn
        total_weighted_recall_sum += w_recall
        total_grounding_rate_sum += case_grounding
        valid_cases_count += 1

        # Accumulate new metrics
        if "total_h_recall_sum" not in locals(): total_h_recall_sum = 0.0
        total_h_recall_sum += h_recall

        if "total_fdr_sum" not in locals(): total_fdr_sum = 0.0
        total_fdr_sum += fdr

        # Store Result
        case_result = {
            "filename": filename,
            "duration_sec": round(duration, 3),
            "flags_found_count": len(report.flags),
            "tp": case_tp,
            "fp": case_fp,
            "fn": case_fn,
            "metrics": {
                "f1": round(f1, 2),
                "precision": round(precision, 2),
                "recall": round(recall, 2),
                "weighted_recall": round(w_recall, 2),
                "high_severity_recall": round(h_recall, 2),
                "fdr": round(fdr, 2),
                "grounding_rate": round(case_grounding, 2)
            }
        }
        results["cases"].append(case_result)
        print(f"Processed {filename}: F1={f1:.2f} W-Recall={w_recall:.2f} H-Recall={h_recall:.2f}")

    # Finalize Aggregates
    if valid_cases_count > 0:
        avg_runtime = total_runtime / valid_cases_count
        macro_w_recall = total_weighted_recall_sum / valid_cases_count
        macro_grounding = total_grounding_rate_sum / valid_cases_count
        macro_h_recall = total_h_recall_sum / valid_cases_count
        macro_fdr = total_fdr_sum / valid_cases_count

        # Micro-averaged F1 (across all items) or Macro?
        # Requirement was simple, let's provide dataset-wide metrics based on total counts
        dataset_precision, dataset_recall, dataset_f1 = precision_recall_f1(total_tp, total_fp, total_fn)
    else:
        avg_runtime = 0
        macro_w_recall = 0
        macro_grounding = 0
        macro_h_recall = 0
        macro_fdr = 0
        dataset_precision = 0
        dataset_recall = 0
        dataset_f1 = 0

    results["summary"] = {
        "total_cases": valid_cases_count,
        "avg_runtime_sec": round(avg_runtime, 3),
        "f1": round(dataset_f1, 3),
        "precision": round(dataset_precision, 3),
        "recall": round(dataset_recall, 3),
        "weighted_recall": round(macro_w_recall, 3),
        "high_severity_recall": round(macro_h_recall, 3),
        "avg_fpr_fdr": round(macro_fdr, 3),
        "grounding_rate": round(macro_grounding, 3)
    }

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    generate_markdown_report(results)
    print(f"Evaluation Complete. Aggregated F1: {dataset_f1:.2f}")

    # Validation Rule: Clean Cases must have 0 flags
    validate_no_false_flags_for_clean_cases(results)

def validate_no_false_flags_for_clean_cases(results: Dict):
    """
    Asserts that cases 004-007 (Clean cases) produced ZERO flags.
    Raises RuntimeError if this safety invariant is violated.
    """
    clean_cases = ["case_004.json", "case_005.json", "case_006.json", "case_007.json",
                   "case_016.json", "case_017.json", "case_018.json"]
    violations = []

    for case in results["cases"]:
        if case["filename"] in clean_cases:
            if case["flags_found_count"] > 0:
                violations.append(f"{case['filename']} (Found {case['flags_found_count']} flags)")

    if violations:
        error_msg = f"SAFETY INVARIANT FAILED: Clean cases produced false positives!\nViolations: {', '.join(violations)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    else:
        print("✅ Safety Invariant Passed: All clean cases produced 0 flags.")



if __name__ == "__main__":
    run_eval_pipeline()
