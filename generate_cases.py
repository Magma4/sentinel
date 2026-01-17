import json
import os

cases = []

# Case 008: Allergy Mismatch (Sulfa)
cases.append({
    "filename": "case_008.json",
    "data": {
        "patient_id": "CASE-008",
        "meds": ["Bactrim"],
        "allergies": ["Sulfa Drugs"],
        "conditions": ["UTI"],
        "labs": [],
        "notes": [
            {
                "date": "2024-05-10",
                "author": "Dr. Uro",
                "content": "Patient presents with dysuria. Dipstick positive.\nAllergies: Sulfa (hives).\nPlan: Start Bactrim DS BID x 3 days."
            }
        ],
        "ground_truth": [
            {
                "category": "MED_LAB_CONFLICT",
                "severity": "HIGH",
                "key": "Sulfa",
                "explanation": "Bactrim contains sulfa, patient has documented sulfa allergy."
            }
        ]
    }
})

# Case 009: Duplicate Therapy (NSAIDs)
cases.append({
    "filename": "case_009.json",
    "data": {
        "patient_id": "CASE-009",
        "meds": ["Ibuprofen", "Naproxen"],
        "allergies": [],
        "conditions": ["Back Pain"],
        "labs": [],
        "notes": [
            {
                "date": "2024-05-12",
                "author": "Dr. Ortho",
                "content": "Chronic low back pain.\nCurrent Meds: Ibuprofen 800mg TID.\nPlan: Adding Naproxen 500mg BID for breakthrough pain."
            }
        ],
        "ground_truth": [
            {
                "category": "MED_LAB_CONFLICT",
                "severity": "MEDIUM",
                "key": "Duplicate NSAID",
                "explanation": "Concurrent use of Ibuprofen and Naproxen increases GI bleed risk."
            }
        ]
    }
})

# Case 010: Missing Follow-up (Abnormal TSH)
cases.append({
    "filename": "case_010.json",
    "data": {
        "patient_id": "CASE-010",
        "meds": [],
        "allergies": [],
        "conditions": ["Fatigue"],
        "labs": [
            {"name": "TSH", "value": 15.4, "unit": "mIU/L", "date": "2024-05-01"}
        ],
        "notes": [
            {
                "date": "2024-05-01",
                "author": "Dr. GP",
                "content": "Reviewing labs. TSH 15.4 (High). Lipids normal.\nPlan: Follow up in 1 year for physical."
            }
        ],
        "ground_truth": [
            {
                "category": "MISSING_WORKFLOW_STEP",
                "severity": "HIGH",
                "key": "TSH High",
                "explanation": "Markedly elevated TSH (15.4) ignored; no thyroid replacement or follow-up ordered."
            }
        ]
    }
})

# Case 011: Temporal Contradiction
cases.append({
    "filename": "case_011.json",
    "data": {
        "patient_id": "CASE-011",
        "meds": ["Albuterol"],
        "allergies": [],
        "conditions": ["Asthma"],
        "labs": [
             {"name": "O2 Sat", "value": 88, "unit": "%", "date": "2024-05-15"}
        ],
        "notes": [
            {
                "date": "2024-05-15",
                "author": "Dr. Pulm",
                "content": "Asthma check. Patient reports doing great, no shortness of breath.\nVitals: O2 Sat 88% on room air.\nPlan: Continue current management. See in 6 months."
            }
        ],
        "ground_truth": [
            {
                "category": "TEMPORAL_CONTRADICTION",
                "severity": "HIGH",
                "key": "Hypoxia",
                "explanation": "Note claims patient doing well but O2 Sat is critical (88%)."
            }
        ]
    }
})

# Case 012: Documentation Inconsistency
cases.append({
    "filename": "case_012.json",
    "data": {
        "patient_id": "CASE-012",
        "meds": [],
        "allergies": [],
        "conditions": [],
        "labs": [],
        "notes": [
            {
                "date": "2024-05-20",
                "author": "Dr. Surg",
                "content": "Procedure Note: Right Knee Injection.\nSite: Left knee prepped and draped.\nInjected 40mg Kenalog into joint space."
            }
        ],
        "ground_truth": [
            {
                "category": "DOC_INCONSISTENCY",
                "severity": "MEDIUM",
                "key": "Laterality",
                "explanation": "Procedure title says Right Knee but body describes Left Knee prep."
            }
        ]
    }
})

# Case 013: Med-Lab Risk (Warfarin/INR)
cases.append({
    "filename": "case_013.json",
    "data": {
        "patient_id": "CASE-013",
        "meds": ["Warfarin"],
        "allergies": [],
        "conditions": ["AFib"],
        "labs": [
            {"name": "INR", "value": 5.2, "unit": "", "date": "2024-05-22"}
        ],
        "notes": [
            {
                "date": "2024-05-22",
                "author": "Dr. Cardio",
                "content": "Routine INR check. Result 5.2.\nPlan: Continue current Warfarin dose 5mg daily. Recheck 4 weeks."
            }
        ],
        "ground_truth": [
            {
                "category": "MED_LAB_CONFLICT",
                "severity": "HIGH",
                "key": "High INR",
                "explanation": "Supratherapeutic INR (5.2) requires dose hold/reduction, not continuation."
            }
        ]
    }
})

# Case 014: Multiple (Allergy + Duplicate)
cases.append({
    "filename": "case_014.json",
    "data": {
        "patient_id": "CASE-014",
        "meds": ["Aspirin", "Ibuprofen"],
        "allergies": ["Aspirin"],
        "conditions": ["Arthritis"],
        "labs": [],
        "notes": [
            {
                "date": "2024-05-25",
                "author": "Dr. Rheum",
                "content": "Pain management. Taking Ibuprofen OTC.\nAllergies: Aspirin (Anaphylaxis).\nPlan: Start Aspirin 81mg for heart health."
            }
        ],
        "ground_truth": [
             {
                "category": "MED_LAB_CONFLICT",
                "severity": "HIGH",
                "key": "Aspirin Allergy",
                "explanation": "Prescribing Aspirin despite anaphylactic allergy."
            },
            {
                "category": "MED_LAB_CONFLICT",
                "severity": "MEDIUM",
                "key": "Duplicate NSAID",
                "explanation": "Aspirin + Ibuprofen significantly increases bleed risk."
            }
        ]
    }
})

# Case 015: Multiple (Med-Lab + Workflow)
cases.append({
    "filename": "case_015.json",
    "data": {
        "patient_id": "CASE-015",
        "meds": ["Simvastatin", "Gemfibrozil"],
        "allergies": [],
        "conditions": ["Hyperlipidemia"],
        "labs": [
            {"name": "ALT", "value": 150, "unit": "U/L", "date": "2024-05-28"},
            {"name": "AST", "value": 140, "unit": "U/L", "date": "2024-05-28"}
        ],
        "notes": [
            {
                "date": "2024-05-28",
                "author": "Dr. Lipid",
                "content": "Follow up. ALT/AST elevated (150/140).\nPlan: Continue Simvastatin and Gemfibrozil."
            }
        ],
        "ground_truth": [
             {
                "category": "MED_LAB_CONFLICT",
                "severity": "HIGH",
                "key": "Statin-Fibrate Interaction",
                "explanation": "Simvastatin + Gemfibrozil interaction risk (Rhabdo)."
            },
            {
                "category": "MED_LAB_CONFLICT",
                "severity": "HIGH",
                "key": "Liver Injury",
                "explanation": "Continuing hepatotoxic meds despite 3x elevation in LFTs."
            }
        ]
    }
})

output_dir = "data/synthetic"
for case in cases:
    path = os.path.join(output_dir, case["filename"])
    with open(path, "w") as f:
        json.dump(case["data"], f, indent=2)
    print(f"Generated {path}")
