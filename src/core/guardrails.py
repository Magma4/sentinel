from typing import List
from .schema import SafetyReport, SafetyFlag

UNSAFE_VERBS = {"start", "stop", "discontinue", "order", "administer"}
CAUTIOUS_WORDS = {"may", "could", "consider", "possible", "potential", "review", "verify", "evaluate", "monitor"}

def contains_unsafe_action_language(text: str) -> bool:
    """Checks if text contains unsafe directive verbs (imperative actions)."""
    if not text:
        return False

    tokens = set(text.lower().replace(".", "").replace(",", "").split())
    return bool(UNSAFE_VERBS.intersection(tokens))

def contains_required_uncertainty_language(text: str) -> bool:
    """Checks if text contains required cautious/advisory language."""
    if not text:
        return False

    tokens = set(text.lower().replace(".", "").replace(",", "").split())
    return bool(CAUTIOUS_WORDS.intersection(tokens))

def sanitize_text(text: str) -> str:
    """Replaces unsafe verbs with neutral alternatives (e.g. "Order" -> "Review")."""
    if not text:
        return text

    sanitized = text
    lower_text = text.lower()

    for term in UNSAFE_VERBS:
        if term in lower_text:
             import re
             pattern = re.compile(re.escape(term), re.IGNORECASE)
             sanitized = pattern.sub("Review", sanitized)

    return sanitized

def validate_report_guardrails(report: SafetyReport) -> SafetyReport:
    """
    Validates and performs post-hoc sanitization on the Safety Report.
    Ensures no clinical commands are present and evidence is cited.
    """
    violations = []

    for index, flag in enumerate(report.flags):
        # 1. Sanitize Explanation
        if contains_unsafe_action_language(flag.explanation):
            flag.explanation = sanitize_text(flag.explanation)

        # 2. Sanitize Recommendation
        if flag.recommendation and contains_unsafe_action_language(flag.recommendation):
            flag.recommendation = sanitize_text(flag.recommendation)

        # 3. Enforce Cautious Tone
        text_to_check = f"{flag.explanation} {flag.recommendation or ''}"
        if not contains_required_uncertainty_language(text_to_check):
             flag.explanation = "Review Item: " + flag.explanation

        # 4. Evidence Constraint
        if not flag.evidence:
             violations.append(f"Flag {index} ({flag.category}) has no evidence.")

    if violations:
        raise ValueError(f"Guardrail Violations:\n" + "\n".join(violations))

    return report
