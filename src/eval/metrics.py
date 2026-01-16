from typing import List, Dict, Any, Tuple
from src.core.schema import SafetyFlag, GroundTruthItem, SafetySeverity

SEVERITY_WEIGHTS = {
    "HIGH": 3.0,
    "MEDIUM": 2.0,
    "LOW": 1.0,
    SafetySeverity.HIGH: 3.0,
    SafetySeverity.MEDIUM: 2.0,
    SafetySeverity.LOW: 1.0
}

def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calculates standard classification metrics."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def severity_weighted_recall(matched_gt: List[GroundTruthItem], all_gt: List[GroundTruthItem]) -> float:
    """Calculates recall weighted by severity (HIGH=3x, MED=2x)."""
    if not all_gt:
        return 1.0 # Trivial success

    total_weight = sum([SEVERITY_WEIGHTS.get(item.severity, 1.0) for item in all_gt])
    recovered_weight = sum([SEVERITY_WEIGHTS.get(item.severity, 1.0) for item in matched_gt])

    return recovered_weight / total_weight if total_weight > 0 else 0.0

def evidence_grounding_rate(flag: SafetyFlag, note: str, labs: str, meds: str) -> bool:
    """Returns True if ALL evidence quotes for this flag exist in the source texts."""
    if not flag.evidence:
        return False

    full_text = (note + "\n" + labs + "\n" + meds).lower()

    for ev in flag.evidence:
        quote = ev.quote

        # Handle simplified source prefixes if present
        if quote.startswith("[") and "] " in quote:
            _, content = quote.split("] ", 1)
            to_search = content.lower()
        else:
            to_search = quote.lower()

        if to_search not in full_text:
            return False
    return True

def high_severity_recall(matched_gt: List[GroundTruthItem], all_gt: List[GroundTruthItem]) -> float:
    """Calculates recall specifically for HIGH severity items."""
    high_sev_gt = [gt for gt in all_gt if gt.severity == "HIGH" or gt.severity == SafetySeverity.HIGH]
    if not high_sev_gt:
        return 1.0

    matched_high = [gt for gt in matched_gt if gt in high_sev_gt]

    return len(matched_high) / len(high_sev_gt)
