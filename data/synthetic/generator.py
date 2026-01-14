import random
from src.core.models import PatientRecord, ClinicalNote

def generate_patient_record() -> PatientRecord:
    # Scenario: Penicillin Allergy with Amoxicillin Prescription
    return PatientRecord(
        patient_id="SYN-001",
        age=45,
        gender="Male",
        allergies=["Penicillin", "Peanuts"],
        conditions=["Hypertension"],
        medications=["Lisinopril 10mg", "Amoxicillin 500mg"],
        vitals={"BP": "140/90", "HR": "88", "Temp": "38.5C"},
        notes=[
            ClinicalNote(
                date="2024-01-14",
                author="Dr. Smith",
                content="Patient presents with high fever and sore throat. Strep test positive. Starting Amoxicillin."
            )
        ]
    )

def generate_safe_record() -> PatientRecord:
    return PatientRecord(
        patient_id="SYN-002",
        age=32,
        gender="Female",
        allergies=[],
        conditions=["Migraine"],
        medications=["Sumatriptan"],
        vitals={"BP": "120/80", "HR": "72", "Temp": "37.0C"},
        notes=[
            ClinicalNote(
                date="2024-01-14",
                author="Dr. Jones",
                content="Patient complains of headache. Neuro exam normal."
            )
        ]
    )
