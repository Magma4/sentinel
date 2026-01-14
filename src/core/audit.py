from typing import List
from .schema import PatientRecord, SafetyReport, SafetyObservation, SafetySeverity, SafetyCategory

class SafetyAuditor:
    """
    Audits patient records for safety risks, inconsistencies, and missing workflow steps.
    Currently uses rule-based logic (Phase 1).
    """

    def audit(self, record: PatientRecord) -> SafetyReport:
        observations = []

        # Rule 1: Penicillin Allergy Check
        if "Penicillin" in record.allergies:
            for med in record.medications:
                if "Amoxicillin" in med or "Penicillin" in med:
                    observations.append(SafetyObservation(
                        category=SafetyCategory.RISK,
                        severity=SafetySeverity.HIGH,
                        evidence=f"Allergies: {record.allergies}, Medications: {med}",
                        explanation="Patient has a documented Penicillin allergy but is prescribed a Penicillin-class antibiotic.",
                        recommendation="Verify allergy status and consider alternative antibiotics."
                    ))

        # Rule 2: High Fever Sepsis Risk (Simple Keyword Check)
        for note in record.notes:
            content_lower = note.content.lower()
            if "high fever" in content_lower and "sepsis" not in content_lower:
                 # Extract snippet for evidence
                 start = max(0, content_lower.find('high fever') - 20)
                 end = min(len(note.content), content_lower.find('high fever') + 40)
                 snippet = note.content[start:end]

                 observations.append(SafetyObservation(
                        category=SafetyCategory.RISK,
                        severity=SafetySeverity.MEDIUM,
                        evidence=f"...{snippet}...",
                        explanation="High fever noted without explicit sepsis screening consideration.",
                        recommendation="Review for sepsis risk factors."
                    ))

        summary = f"Safety audit complete. Found {len(observations)} observations."

        return SafetyReport(
            patient_id=record.patient_id,
            observations=observations,
            summary=summary
        )
