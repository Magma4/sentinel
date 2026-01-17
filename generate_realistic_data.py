import json
import os

output_dir = "data/synthetic"
os.makedirs(output_dir, exist_ok=True)

cases = []

# Case 1: Cardiology Discharge (MRN_4592)
# Scenario: Discharging post-MI patient without Beta Blocker.
cases.append({
    "filename": "Pt_Record_MRN_4592_Cardio_Discharge.json",
    "data": {
        "patient_id": "MRN-4592",
        "meds": ["Aspirin", "Atorvastatin", "Lisinopril"],
        "allergies": [],
        "conditions": ["Acute Myocardial Infarction (STEMI)", "Hypertension"],
        "labs": [
            {"name": "Troponin I", "value": 12.5, "unit": "ng/mL", "date": "2024-06-01"},
            {"name": "LVEF", "value": 40, "unit": "%", "date": "2024-06-02"}
        ],
        "notes": [
            {
                "date": "2024-06-05",
                "author": "Dr. Hart (Cardiology)",
                "content": "Discharge Summary.\nPrincipal Diagnosis: Anterior STEMI.\nIntervention: DES to LAD.\nEcho: LVEF 40%.\nDischarge Plan: Home conformtable. Continue Aspirin, Statin, ACEi. Follow up 2 weeks."
            }
        ],
        "ground_truth": [
            {
                "category": "MISSING_THERAPY",
                "severity": "HIGH",
                "key": "Missing Beta Blocker",
                "explanation": "Patient with recent MI and reduced LVEF (40%) should be on a Beta Blocker unless contraindicated."
            }
        ]
    }
})

# Case 2: Oncology Clinic (MRN_8821)
# Scenario: Neutropenic fever risk ignored.
cases.append({
    "filename": "Pt_Record_MRN_8821_Onc_Clinic.json",
    "data": {
        "patient_id": "MRN-8821",
        "meds": ["Doxorubicin", "Cyclophosphamide"],
        "allergies": ["Penicillin"],
        "conditions": ["Breast Cancer"],
        "labs": [
            {"name": "WBC", "value": 0.8, "unit": "K/uL", "date": "2024-06-10"},
            {"name": "ANC", "value": 200, "unit": "cells/uL", "date": "2024-06-10"},
            {"name": "Temp", "value": 101.5, "unit": "F", "date": "2024-06-10"}
        ],
        "notes": [
            {
                "date": "2024-06-10",
                "author": "NP Smith",
                "content": "Chemo Cycle 3 Day 10. Patient reports feeling warm/chills today.\nExam: T 101.5.\nAssessment: Viral syndrome likely.\nPlan: Tylenol and fluids. Return if worse."
            }
        ],
        "ground_truth": [
            {
                "category": "CLINICAL_RISK",
                "severity": "CRITICAL",
                "key": "Neutropenic Fever",
                "explanation": "Febrile Neutropenia (ANC < 500 + Fever) is a medical emergency requiring broad-spectrum antibiotics, not just Tylenol."
            }
        ]
    }
})

# Case 3: ED Encounter (MRN_1102)
# Scenario: Contrast CT ordered in patient with AKI.
cases.append({
    "filename": "Pt_Record_MRN_1102_ED_Visit.json",
    "data": {
        "patient_id": "MRN-1102",
        "meds": ["Metformin"],
        "allergies": ["Shellfish"],
        "conditions": ["Diabetes", "Abdominal Pain"],
        "labs": [
            {"name": "Creatinine", "value": 2.8, "unit": "mg/dL", "date": "2024-06-12"},
            {"name": "eGFR", "value": 25, "unit": "mL/min", "date": "2024-06-12"}
        ],
        "notes": [
            {
                "date": "2024-06-12",
                "author": "Dr. Swift (ED)",
                "content": "Patient presents with RLQ pain. Rule out Appendicitis.\nLabs: Cr 2.8 (Baseline 1.0).\nPlan: Order CT Abdomen/Pelvis WITH IV Contrast to visualize appendix."
            }
        ],
        "ground_truth": [
            {
                "category": "CONTRAINDICATION",
                "severity": "HIGH",
                "key": "Contrast Nephropathy Risk",
                "explanation": "Ordering IV Contrast in patient with acute renal failure (Cr 2.8, GFR 25) risks worsening kidney injury (CIN). Non-contrast or Ultrasound preferred."
            }
        ]
    }
})

# Case 4: Fracture Clinic (MRN_7734)
# Scenario: NSAID prescribed to patient on Warfarin.
cases.append({
    "filename": "Pt_Record_MRN_7734_Ortho_Fracture.json",
    "data": {
        "patient_id": "MRN-7734",
        "meds": ["Warfarin 5mg daily", "Lisinopril 10mg daily"],
        "allergies": [],
        "conditions": ["Atrial Fibrillation", "Distal Radius Fracture"],
        "labs": [
            {"name": "INR", "value": 2.8, "unit": "", "date": "2025-01-10"},
            {"name": "Creatinine", "value": 0.9, "unit": "mg/dL", "date": "2025-01-10"}
        ],
        "notes": [
            {
                "date": "2025-01-15",
                "author": "Dr. S. Bones",
                "content": "Subjective: 65yo F presents to ED after slip and fall on ice. Complains of 9/10 pain in right wrist.\nObjective: Right distal radius deformed, swollen, tender.\nPlan: \n1. Ibuprofen 600mg PO q6h for pain.\n2. Reduction and casting.\n3. Discharge home."
            }
        ],
        "ground_truth": [
            {
                "category": "MED_MED_CONFLICT",
                "severity": "HIGH",
                "key": "Warfarin + NSAID Interaction",
                "explanation": "Prescribing Ibuprofen (NSAID) to a patient on Warfarin significantly increases bleeding risk. Tylenol is preferred."
            }
        ]
    }
})

# Case 5: ICU Sepsis (MRN_9901)
# Scenario: Severe Sepsis missed.
cases.append({
    "filename": "Pt_Record_MRN_9901_ICU_Sepsis.json",
    "data": {
        "patient_id": "MRN-9901",
        "meds": ["Piperacillin-Tazobactam"],
        "allergies": [],
        "conditions": ["Pneumonia", "Sepsis"],
        "labs": [
            {"name": "Lactate", "value": 4.5, "unit": "mmol/L", "date": "2024-07-01"},
            {"name": "WBC", "value": 22.0, "unit": "K/uL", "date": "2024-07-01"},
            {"name": "BP", "value": "85/50", "unit": "mmHg", "date": "2024-07-01"}
        ],
        "notes": [
            {
                "date": "2024-07-01",
                "author": "Dr. Intensivist",
                "content": "Admit from ED. Hypoxic, febrile.\nLactate elevated at 4.5.\nPlan: Zosyn started. Maintenance fluids at 75cc/hr. Monitor BP."
            }
        ],
        "ground_truth": [
            {
                "category": "CLINICAL_RISK",
                "severity": "CRITICAL",
                "key": "Sepsis Bundle Non-Compliance",
                "explanation": "Elevated Lactate (>4) and hypotension requires 30mL/kg bolus resuscitation immediately, not just maintenance fluids."
            }
        ]
    }
})

# Case 6: Neurology Stroke (MRN_2255)
# Scenario: tPA considered but BP > 185/110.
cases.append({
    "filename": "Pt_Record_MRN_2255_Stroke_Code.json",
    "data": {
        "patient_id": "MRN-2255",
        "meds": ["Amlodipine"],
        "allergies": [],
        "conditions": ["Acute Ischemic Stroke"],
        "labs": [
            {"name": "Glucose", "value": 110, "unit": "mg/dL", "date": "2024-07-05"},
            {"name": "BP", "value": "195/110", "unit": "mmHg", "date": "2024-07-05"}
        ],
        "notes": [
            {
                "date": "2024-07-05",
                "author": "Dr. Neuro",
                "content": "Code Stroke. LKW 2 hours ago. NIHSS 12.\nCT Head negative for bleed.\nBP 195/110.\nPlan: Proceed with IV tPA bolus immediately."
            }
        ],
        "ground_truth": [
            {
                "category": "CONTRAINDICATION",
                "severity": "CRITICAL",
                "key": "tPA Hypertension",
                "explanation": "tPA cannot be given if BP > 185/110 due to bleed risk. BP must be lowered with Labetalol/Nicardipine first."
            }
        ]
    }
})

# Case 7: Renal Function (MRN_031) [Migrated from scenario_01_renal]
# Scenario: Metformin + CKD.
cases.append({
    "filename": "Pt_Record_MRN_031_Renal.json",
    "data": {
        "patient_id": "CASE-031",
        "note": "65M with T2DM, HTN. Here for routine follow-up. BP 135/85. Creatinine has trended up over past 6 months. Plan: Continue current regimen including Metformin.",
        "labs": [
            {"name": "Creatinine", "value": 1.7, "unit": "mg/dL", "date": "2024-01-20"},
            {"name": "eGFR", "value": 38, "unit": "mL/min", "date": "2024-01-20"},
            {"name": "HbA1c", "value": 7.2, "unit": "%", "date": "2024-01-20"}
        ],
        "meds": ["Metformin 1000mg BID", "Lisinopril 20mg daily"],
        "ground_truth": [
            {"category": "MED_LAB_CONFLICT", "severity": "HIGH", "key": "Metformin Renal"}
        ]
    }
})

# Case 8: Potassium (MRN_032) [Migrated from scenario_02_potassium]
# Scenario: Hyperkalemia discharge.
cases.append({
    "filename": "Pt_Record_MRN_032_Potassium.json",
    "data": {
        "patient_id": "CASE-032",
        "note": "78F admitted for weakness. Labs showed hyperkalemia. ECG normal. Feeling better today. Discharging home to follow up with PCP in 1 week.",
        "labs": [
            {"name": "Potassium", "value": 6.1, "unit": "mmol/L", "date": "2024-02-10"}
        ],
        "meds": ["Spironolactone 25mg daily"],
        "ground_truth": [
            {"category": "MISSING_WORKFLOW_STEP", "severity": "HIGH", "key": "Hyperkalemia Discharge"}
        ]
    }
})

# Case 9: Penicillin (MRN_033) [Migrated from scenario_03_penicillin]
# Scenario: Allergy.
cases.append({
    "filename": "Pt_Record_MRN_033_Penicillin.json",
    "data": {
        "patient_id": "CASE-033",
        "note": "Subjective: Sore throat, fever. Objective: Tonsillar exudates. Hx: Severe hives with Penicillin. Assessment: Strep Pharyngitis. Plan: Start Amoxicillin 500mg TID x 10 days.",
        "labs": [
            {"name": "Rapid Strep", "value": "Positive", "unit": "N/A", "date": "2024-03-01"}
        ],
        "meds": ["Amoxicillin 500mg TID"],
        "ground_truth": [
            {"category": "MED_LAB_CONFLICT", "severity": "HIGH", "key": "Penicillin Allergy"}
        ]
    }
})


for case in cases:
    path = os.path.join(output_dir, case["filename"])
    with open(path, "w") as f:
        json.dump(case["data"], f, indent=2)
    print(f"Generated {path}")
