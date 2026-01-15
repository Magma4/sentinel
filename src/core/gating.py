import re
from typing import List, Dict, Any
from .schema import SafetyReport, SafetyFlag, SafetyCategory, SafetySeverity

def has_numeric_value(text: str) -> bool:
    """Checks if text contains a numeric digit."""
    return bool(re.search(r'\d', text))

def gate_safety_flags(report: SafetyReport) -> SafetyReport:
    """
    Filters out low-quality flags based on:
    - Confidence threshold
    - Evidence quality (number of quotes)
    - Severity-specific rules
    """
    if not report.metadata:
        report.metadata = {}

    if not report.flags:
        report.metadata["gating_decisions"] = []
        return report

    kept_flags = []
    dropped_flags = []

    for flag in report.flags:
        # Rule 1: Minimum confidence
        if flag.confidence < 0.5:
            dropped_flags.append({"flag": flag.explanation[:50], "reason": f"Low confidence: {flag.confidence}"})
            continue

        # Rule 2: Evidence requirement
        if not flag.evidence or len(flag.evidence) == 0:
            dropped_flags.append({"flag": flag.explanation[:50], "reason": "No evidence"})
            continue

        # Rule 3: Severity-specific thresholds
        if flag.severity == SafetySeverity.HIGH and flag.confidence < 0.65:
            dropped_flags.append({"flag": flag.explanation[:50], "reason": f"HIGH severity needs >0.65 confidence, got {flag.confidence}"})
            continue

        kept_flags.append(flag)

    # Store gating decisions in metadata
    if not report.metadata:
        report.metadata = {}
    report.metadata["gating_kept"] = len(kept_flags)
    report.metadata["gating_dropped"] = len(dropped_flags)
    report.metadata["gating_decisions"] = dropped_flags

    return SafetyReport(
        patient_id=report.patient_id,
        summary=report.summary,
        flags=kept_flags,
        missing_info_questions=report.missing_info_questions,
        metadata=report.metadata
    )


def calibrate_confidence(report: SafetyReport) -> SafetyReport:
    """
    Adjusts confidence scores based on heuristic rules regarding severity and evidence strength.

    Rules:
    1. HIGH severity + >= 2 evidence quotes -> Boost to min 0.80 (Strong evidence warrants high trust)
    2. MEDIUM severity + <= 1 evidence quote -> Cap at 0.65 (Weak evidence limits trust)
    3. LOW severity -> Cap at 0.55 (Minor issues should not be overconfident)
    """
    for flag in report.flags:
        # Rule 1: High Severity Boost
        # If it's a critical safety issue and we have multiple pieces of evidence (e.g. Note + Meds),
        # we boost confidence to ensure it surfaces prominently.
        if flag.severity == SafetySeverity.HIGH and len(flag.evidence) >= 2:
            if flag.confidence < 0.80:
                flag.confidence = 0.80

        # Rule 2: Medium Severity Cap (Weak Evidence)
        # If a medium issue has scant evidence (only 1 quote), we limit confidence
        # to prevent "alert fatigue" from potentially ambiguous findings.
        elif flag.severity == SafetySeverity.MEDIUM and len(flag.evidence) <= 1:
            if flag.confidence > 0.65:
                flag.confidence = 0.65

        # Rule 3: Low Severity Cap
        # Low severity items are mostly informational; model shouldn't claim certainty.
        elif flag.severity == SafetySeverity.LOW:
            if flag.confidence > 0.55:
                flag.confidence = 0.55

    return report
