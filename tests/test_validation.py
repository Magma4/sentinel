import pytest
from pydantic import ValidationError
from src.core.schema import SafetyFlag, SafetyCategory, SafetySeverity, Evidence

def test_unsafe_language_rejection():
    """
    Test that SafetyFlag rejects unsafe directive language.
    """
    with pytest.raises(ValidationError) as excinfo:
        SafetyFlag(
            category=SafetyCategory.MED_LAB_CONFLICT,
            severity=SafetySeverity.HIGH,
            confidence=1.0,
            evidence=[Evidence(quote="test")],
            explanation="Unsafe",
            recommendation="You should prescribe more meds."
        )
    assert "Unsafe directive language detected" in str(excinfo.value)
    assert "prescribe" in str(excinfo.value)

def test_safe_language_allowed():
    """
    Test that SafetyFlag allows safe advisory language.
    """
    flag = SafetyFlag(
        category=SafetyCategory.MED_LAB_CONFLICT,
        severity=SafetySeverity.HIGH,
        confidence=1.0,
        evidence=[Evidence(quote="test")],
        explanation="Safe observation",
        recommendation="Consider reviewing the medication list."
    )
    assert flag.recommendation == "Consider reviewing the medication list."
