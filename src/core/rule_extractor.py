"""
Simple rule-based fact extraction for One-Call Demo Mode.
Extracts minimal facts using regex/pattern matching instead of LLM.
"""
import re
from typing import Dict, Any, List


def extract_facts_rule_based(note: str, labs: str, meds: str) -> Dict[str, Any]:
    """
    Fast, rule-based fact extraction without LLM.
    Returns minimal facts_json compatible with safety audit.
    """
    facts = {
        "patient_age": "UNKNOWN",
        "sex": "UNKNOWN",
        "key_conditions": [],
        "current_meds": [],
        "allergies": [],
        "labs": [],
        "clinician_assertions": []
    }

    # Extract allergies
    allergy_patterns = [
        r"allerg(?:y|ies)?\s*(?:to\s*)?:?\s*([^\n]+)",
        r"nkda",
        r"no known (?:drug )?allergies"
    ]

    for pattern in allergy_patterns:
        matches = re.findall(pattern, note.lower())
        if matches:
            if isinstance(matches[0], str) and matches[0].strip():
                facts["allergies"].append(matches[0].strip().title())
        elif "nkda" in pattern or "no known" in pattern:
            if re.search(pattern, note.lower()):
                facts["allergies"].append("NKDA")
                break

    # Extract medications
    if meds:
        med_lines = [line.strip() for line in meds.split('\n') if line.strip()]
        facts["current_meds"] = med_lines[:10]  # Limit to 10

    # Extract key labs
    lab_patterns = {
        "eGFR": r"egfr[:\s]+([0-9\.]+)",
        "Creatinine": r"creat(?:inine)?[:\s]+([0-9\.]+)",
        "Potassium": r"(?:potassium|k)[:\s]+([0-9\.]+)"
    }

    for lab_name, pattern in lab_patterns.items():
        match = re.search(pattern, labs.lower())
        if match:
            facts["labs"].append({
                "name": lab_name,
                "value": match.group(1),
                "unit": "mg/dL" if "Creat" in lab_name else ("mEq/L" if "Potassium" in lab_name else ""),
                "date": "UNKNOWN"
            })

    # Extract clinician assertions
    assertion_keywords = ["stable", "improving", "worsening", "worsen", "deteriorat"]
    sentences = re.split(r'[.!?]\s+', note)

    for sentence in sentences:
        if any(kw in sentence.lower() for kw in assertion_keywords):
            facts["clinician_assertions"].append(sentence.strip())

    return facts
