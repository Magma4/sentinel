from typing import List, Dict, Any, Tuple
from src.core.schema import SafetyFlag, GroundTruthItem, SafetySeverity

SEVERITY_WEIGHTS = {
    "HIGH": 3.0,
    "MEDIUM": 2.0,
    "LOW": 1.0,
    # Fallback/Enum handling
    SafetySeverity.HIGH: 3.0,
    SafetySeverity.MEDIUM: 2.0,
    SafetySeverity.LOW: 1.0
}

def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Calculates Precision, Recall, and F1 score.
    Returns (precision, recall, f1).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def severity_weighted_recall(matched_gt: List[GroundTruthItem], all_gt: List[GroundTruthItem]) -> float:
    """
    Calculates weighted recall based on severity importance.
    HIGH = 3, MEDIUM = 2, LOW = 1.
    """
    if not all_gt:
        return 1.0 # Trivial success if nothing to find

    total_weight = sum([SEVERITY_WEIGHTS.get(item.severity, 1.0) for item in all_gt])
    recovered_weight = sum([SEVERITY_WEIGHTS.get(item.severity, 1.0) for item in matched_gt])

    return recovered_weight / total_weight if total_weight > 0 else 0.0

def evidence_grounding_rate(flag: SafetyFlag, note: str, labs: str, meds: str) -> bool:
    """
    Checks if coverage of evidence is grounded in source texts.
    Returns True if ALL evidence quotes for this flag are found in at least one source text.
    """
    if not flag.evidence:
        return False # Claims without evidence are not grounded

    # Combine sources for simpler check (or check metadata if we tracked source)
    # The flag.evidence quotes usually contain [SOURCE] prefix if built by our new helper,
    # but let's check widely for robustness.
    full_text = (note + "\n" + labs + "\n" + meds).lower()

    for ev in flag.evidence:
        # Strip potential [SOURCE] prefix if we added it, or just search substring
        # Our helper adds "[NOTE] quote", so we might need to be careful.
        # Let's search for the raw quote part if it looks decorated, or just the string.
        quote = ev.quote

        # Simple cleanup for the check: if format is "[SOURCE] content", extract content
        if quote.startswith("[") and "] " in quote:
            _, content = quote.split("] ", 1)
            to_search = content.lower()
        else:
            to_search = quote.lower()

        if to_search not in full_text:
            return False


def high_severity_recall(matched_gt: List[GroundTruthItem], all_gt: List[GroundTruthItem]) -> float:
    """
    Calculates recall ONLY for HIGH severity items.
    """
    high_sev_gt = [gt for gt in all_gt if gt.severity == "HIGH" or gt.severity == SafetySeverity.HIGH]
    if not high_sev_gt:
        return 1.0 # No high severity issues to find, strictly successful coverage

    # helper to check identity or containment
    # matched_gt contains only the items we successfully found
    # We need to intersect matched_gt with high_sev_gt

    # We rely on object identity or key matching.
    # In run_eval, we select matched_gt from all_gt. So identity check works.
    matched_high = [gt for gt in matched_gt if gt in high_sev_gt]

    return len(matched_high) / len(high_sev_gt)
