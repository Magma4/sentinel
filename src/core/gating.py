import re
from typing import List, Dict, Any
from .schema import SafetyReport, SafetyFlag, SafetyCategory, SafetySeverity

def has_numeric_value(text: str) -> bool:
    """Checks if text contains a numeric digit."""
    return bool(re.search(r'\d', text))

def gate_safety_flags(report: SafetyReport) -> SafetyReport:
    """
    Quality gate to filter out low-confidence hallucinations and weak signals.
    Enforces minimum confidence and evidence requirements.
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

        # Rule 4: Strict Evidence Grounding
        # If ALL evidence sources are UNKNOWN, drop as probable hallucination.
        if all(getattr(ev, "source", "UNKNOWN") in ["UNKNOWN", None] for ev in flag.evidence):
             dropped_flags.append({"flag": flag.explanation[:50], "reason": "No grounded evidence found (Hallucination risk)"})
             continue

        kept_flags.append(flag)

    # Telemetry
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
    Heuristically adjusts confidence based on evidence strength and severity.
    - Boosts verified HIGH severity issues.
    - Caps confidence for weak evidence.
    """
    for flag in report.flags:
        # Rule 1: High Severity Boost (Strong Evidence)
        if flag.severity == SafetySeverity.HIGH and len(flag.evidence) >= 2:
            if flag.confidence < 0.80:
                flag.confidence = 0.80

        # Rule 2: Medium Severity Cap (Weak Evidence)
        elif flag.severity == SafetySeverity.MEDIUM and len(flag.evidence) <= 1:
            if flag.confidence > 0.65:
                flag.confidence = 0.65

        # Rule 3: Low Severity Cap (General caution)
        elif flag.severity == SafetySeverity.LOW:
            if flag.confidence > 0.55:
                flag.confidence = 0.55

    return report
