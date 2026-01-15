from typing import List
from .schema import SafetyReport, SafetyFlag

UNSAFE_VERBS = {"start", "stop", "discontinue", "order", "administer"}
CAUTIOUS_WORDS = {"may", "could", "consider", "possible", "potential", "review", "verify", "evaluate", "monitor"}

def contains_unsafe_action_language(text: str) -> bool:
    """
    Checks if text contains unsafe directive verbs (imperative actions).
    """
    if not text:
        return False

    # Simple tokenization for keyword matching
    tokens = set(text.lower().replace(".", "").replace(",", "").split())
    return bool(UNSAFE_VERBS.intersection(tokens))

def contains_required_uncertainty_language(text: str) -> bool:
    """
    Checks if text contains required cautious/advisory language.
    """
    if not text:
        return False # Empty text doesn't contain uncertainty, strictly speaking.
                     # However, if strictness requires it, we might return False.
                     # But usually we check this on fields that exist.

    tokens = set(text.lower().replace(".", "").replace(",", "").split())
    return bool(CAUTIOUS_WORDS.intersection(tokens))

def sanitize_text(text: str) -> str:
    """
    Sanitizes text by finding unsafe verbs and replacing them with neutral alternatives.
    Example: "Result suggests stopping medication" -> "Result [Review: suggests] [Review: stopping] medication" -> "Result indicates checking medication"
    Actually simpler: Just replace the forbidden word with "Review".
    """
    if not text:
        return text

    sanitized = text
    lower_text = text.lower()

    # We iterate over unsafe verbs. If found, we replace.
    # This is a bit naive (case sensitivity), but strict safety prefers over-correction.
    for term in UNSAFE_VERBS:
        # Use regex to replace whole words only to avoid partial matches?
        # For simplicity and strictness, let's just do case-insensitive replace of the word.
        # But we need to preserve the original string case if possible?
        # Actually, replacing "Stop" with "Review" is fine.

        # Simple approach: Replace known forbidden terms with "Review" or related.
        if term in lower_text:
             # Case insensitive replacement using re
             import re
             pattern = re.compile(re.escape(term), re.IGNORECASE)
             sanitized = pattern.sub("Review", sanitized)

    return sanitized

def validate_report_guardrails(report: SafetyReport) -> SafetyReport:
    """
    Validates AND Sanitizes a SafetyReport.
    1. Sanitizes explanation/recommendation text.
    2. Raises error if sanitization fails or other constraints met.
    """
    violations = []

    for index, flag in enumerate(report.flags):
        # Sanitize Explanation
        if contains_unsafe_action_language(flag.explanation):
            flag.explanation = sanitize_text(flag.explanation)

        # Sanitize Recommendation
        if flag.recommendation and contains_unsafe_action_language(flag.recommendation):
            flag.recommendation = sanitize_text(flag.recommendation)

        # Re-Check for Cautious Language (Sanitization usually adds 'Review', which is cautious)
        text_to_check = f"{flag.explanation} {flag.recommendation or ''}"

        # We require at least one cautious term
        if not contains_required_uncertainty_language(text_to_check):
             # Force add cautious preamble if missing
             flag.explanation = "Review Item: " + flag.explanation

        # 3. Evidence Check
        if not flag.evidence:
             violations.append(f"Flag {index} ({flag.category}) has no evidence.")

    if violations:
        raise ValueError(f"Safety Guardrail Violations:\n" + "\n".join(violations))

    return report
