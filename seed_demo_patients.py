"""
Seed Demo Patients — Creates 5 clinically compelling demo patients
for SentinelMD demonstrations and YC demo recordings.

Each patient has a pre-filled encounter with notes, labs, and meds
designed to trigger meaningful safety flags when audited.

Usage:
    python seed_demo_patients.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.patient_service import PatientService

DEMO_PATIENTS = [
    {
        "name": "Maria Santos",
        "dob": "1981-03-15",
        "mrn": "MRN-1001",
        "encounter": {
            "note": (
                "Patient: Maria Santos\n"
                "DOB: 03/15/1981 | MRN: MRN-1001\n\n"
                "Subjective: 45F presents with high fever (38.5°C) and severe sore throat x3 days. "
                "Difficulty swallowing. No cough or rhinorrhea.\n\n"
                "PMH: Hypertension (well-controlled on Lisinopril)\n"
                "Allergies: PENICILLIN (anaphylaxis), Peanuts\n\n"
                "Objective:\n"
                "  Vitals: BP 140/90, HR 88, Temp 38.5°C, SpO2 98%\n"
                "  HEENT: Oropharynx erythematous, tonsillar exudates bilateral\n"
                "  Neck: Tender anterior cervical lymphadenopathy\n\n"
                "Assessment: Strep Pharyngitis (rapid strep positive)\n\n"
                "Plan: Start Amoxicillin 500mg TID x 10 days. "
                "Follow up in 3 days if no improvement."
            ),
            "labs": (
                "Rapid Strep Test: POSITIVE\n"
                "WBC: 14.2 K/uL (H)\n"
                "CRP: 45 mg/L (H)"
            ),
            "meds": (
                "Lisinopril 10mg PO Daily\n"
                "Amoxicillin 500mg PO TID (NEW)"
            ),
        },
    },
    {
        "name": "James Wilson",
        "dob": "1958-11-22",
        "mrn": "MRN-4592",
        "encounter": {
            "note": (
                "Patient: James Wilson\n"
                "DOB: 11/22/1958 | MRN: MRN-4592\n\n"
                "Discharge Summary — Cardiology Service\n\n"
                "Principal Diagnosis: Anterior STEMI\n"
                "Intervention: Drug-eluting stent (DES) to LAD, day 3 post-PCI\n\n"
                "Hospital Course: 68M admitted via EMS with substernal chest pain "
                "and ST-elevation in V1-V4. Emergent cath showed 95% LAD occlusion. "
                "Successful DES placement. Post-procedure course uncomplicated.\n\n"
                "Echo: LVEF 40% with anterior wall hypokinesis\n"
                "Troponin I peaked at 12.5 ng/mL\n\n"
                "Discharge Medications:\n"
                "  - Aspirin 81mg daily\n"
                "  - Atorvastatin 80mg daily\n"
                "  - Lisinopril 5mg daily\n"
                "  - Clopidogrel 75mg daily\n\n"
                "Follow-up: Cardiology in 2 weeks. Cardiac rehab referral placed."
            ),
            "labs": (
                "Troponin I: 12.5 ng/mL (H — peaked)\n"
                "LVEF: 40%\n"
                "LDL: 142 mg/dL (H)\n"
                "Creatinine: 1.1 mg/dL\n"
                "K+: 4.2 mmol/L"
            ),
            "meds": (
                "Aspirin 81mg PO Daily\n"
                "Atorvastatin 80mg PO Daily\n"
                "Lisinopril 5mg PO Daily\n"
                "Clopidogrel 75mg PO Daily"
            ),
        },
    },
    {
        "name": "Sarah Chen",
        "dob": "1954-07-08",
        "mrn": "MRN-9901",
        "encounter": {
            "note": (
                "Patient: Sarah Chen\n"
                "DOB: 07/08/1954 | MRN: MRN-9901\n\n"
                "ICU Admission Note\n\n"
                "72F admitted from ED. Found hypoxic on room air (SpO2 85%), "
                "febrile to 39.2°C, tachycardic HR 112.\n\n"
                "PMH: COPD, Type 2 Diabetes, Hypertension\n"
                "Allergies: None known\n\n"
                "ED Course: CXR shows right lower lobe consolidation. "
                "Blood cultures drawn x2. Lactate 4.5 mmol/L. "
                "BP dipped to 85/50 after 500cc NS bolus.\n\n"
                "Assessment: Severe sepsis secondary to community-acquired pneumonia\n\n"
                "Plan:\n"
                "  - Piperacillin-Tazobactam (Zosyn) 4.5g IV q6h started\n"
                "  - Maintenance IV fluids at 75 cc/hr\n"
                "  - O2 via high-flow nasal cannula\n"
                "  - Monitor BP and urine output hourly\n"
                "  - Recheck lactate in 6 hours"
            ),
            "labs": (
                "Lactate: 4.5 mmol/L (H — CRITICAL)\n"
                "WBC: 22.0 K/uL (H)\n"
                "BP: 85/50 mmHg (post-bolus)\n"
                "Procalcitonin: 8.2 ng/mL (H)\n"
                "Creatinine: 1.8 mg/dL (H)\n"
                "Glucose: 245 mg/dL (H)\n"
                "SpO2: 85% on RA → 94% on HFNC"
            ),
            "meds": (
                "Piperacillin-Tazobactam 4.5g IV q6h\n"
                "Maintenance NS at 75 cc/hr\n"
                "Metformin 1000mg PO BID (home med — held)\n"
                "Lisinopril 20mg PO Daily (home med — held)"
            ),
        },
    },
    {
        "name": "Robert Johnson",
        "dob": "1971-01-30",
        "mrn": "MRN-0031",
        "encounter": {
            "note": (
                "Patient: Robert Johnson\n"
                "DOB: 01/30/1971 | MRN: MRN-0031\n\n"
                "Subjective: 55M presents with progressive fatigue, muscle weakness, "
                "and intermittent palpitations x2 weeks. Denies chest pain or SOB.\n\n"
                "PMH: CKD Stage 3b (eGFR 38), Hypertension, Heart Failure (EF 35%)\n"
                "Allergies: Sulfa drugs\n\n"
                "Objective:\n"
                "  Vitals: BP 155/95, HR 72 (irregular), Temp 37.0°C\n"
                "  CV: Irregularly irregular rhythm, no murmurs\n"
                "  Ext: 1+ bilateral pedal edema\n\n"
                "Assessment: Hyperkalemia in setting of CKD + ACEi + K-sparing diuretic\n\n"
                "Plan:\n"
                "  - Continue current medications\n"
                "  - Recheck BMP in 1 week\n"
                "  - Low-potassium diet counseling"
            ),
            "labs": (
                "Potassium: 6.2 mmol/L (H — CRITICAL)\n"
                "Creatinine: 2.4 mg/dL (H)\n"
                "eGFR: 38 mL/min (L)\n"
                "BUN: 42 mg/dL (H)\n"
                "Magnesium: 1.6 mg/dL (L)"
            ),
            "meds": (
                "Lisinopril 20mg PO Daily\n"
                "Spironolactone 25mg PO Daily\n"
                "Aspirin 81mg PO Daily\n"
                "Furosemide 40mg PO Daily\n"
                "Carvedilol 12.5mg PO BID"
            ),
        },
    },
    {
        "name": "Afia Mensah",
        "dob": "1988-09-12",
        "mrn": "MRN-0033",
        "encounter": {
            "note": (
                "Patient: Afia Mensah\n"
                "DOB: 09/12/1988 | MRN: MRN-0033\n\n"
                "Subjective: 38F presents with sore throat, fever (38.8°C) x2 days. "
                "History of severe hives with Penicillin as a child.\n\n"
                "PMH: Asthma (mild intermittent), Penicillin allergy (urticaria)\n"
                "Allergies: PENICILLIN (severe hives/urticaria)\n\n"
                "Objective:\n"
                "  Vitals: BP 118/72, HR 90, Temp 38.8°C, SpO2 99%\n"
                "  HEENT: Tonsillar exudates, posterior pharyngeal erythema\n"
                "  Lungs: Clear bilaterally, no wheezing\n\n"
                "Assessment: Strep Pharyngitis\n\n"
                "Plan: Start Amoxicillin 500mg TID x 10 days. "
                "Albuterol PRN for asthma. Follow up if worsening."
            ),
            "labs": (
                "Rapid Strep: POSITIVE\n"
                "WBC: 12.8 K/uL (H)\n"
                "CRP: 38 mg/L (H)"
            ),
            "meds": (
                "Amoxicillin 500mg PO TID (NEW)\n"
                "Albuterol 90mcg INH PRN\n"
                "Montelukast 10mg PO Daily"
            ),
        },
    },
    # --- NEW: DDI-demonstrating patients ---
    {
        "name": "Dorothy Kwame",
        "dob": "1946-04-18",
        "mrn": "MRN-2201",
        "encounter": {
            "note": (
                "Patient: Dorothy Kwame\n"
                "DOB: 04/18/1946 | MRN: MRN-2201\n\n"
                "Subjective: 80F presents with right knee pain and swelling x1 week. "
                "Reports stiffness worse in the mornings. No trauma history.\n\n"
                "PMH: Atrial fibrillation (on anticoagulation), Osteoarthritis, "
                "Depression (on SSRI), Hypertension\n"
                "Allergies: None known\n\n"
                "Objective:\n"
                "  Vitals: BP 148/88, HR 78 (irregular), Temp 36.8°C\n"
                "  Right knee: Mild effusion, crepitus, ROM limited by pain\n"
                "  No erythema or warmth suggesting septic joint\n\n"
                "Assessment: Osteoarthritis flare, right knee\n\n"
                "Plan: Start Ibuprofen 600mg TID for 2 weeks for pain relief. "
                "Continue Coumadin and other home medications. "
                "Follow-up in 2 weeks, consider ortho referral."
            ),
            "labs": (
                "INR: 2.8 (therapeutic range 2.0-3.0)\n"
                "Creatinine: 1.2 mg/dL\n"
                "CBC: WNL\n"
                "ESR: 32 mm/hr (H)\n"
                "CRP: 12 mg/L (H)"
            ),
            "meds": (
                "Coumadin 5mg PO Daily\n"
                "Ibuprofen 600mg PO TID (NEW)\n"
                "Zoloft 100mg PO Daily\n"
                "Lisinopril 10mg PO Daily\n"
                "Omeprazole 20mg PO Daily"
            ),
        },
    },
    {
        "name": "Kwesi Appiah",
        "dob": "1975-12-05",
        "mrn": "MRN-3301",
        "encounter": {
            "note": (
                "Patient: Kwesi Appiah\n"
                "DOB: 12/05/1975 | MRN: MRN-3301\n\n"
                "Subjective: 51M presents for chronic low back pain follow-up. "
                "Reports pain 8/10, radiating to left leg. "
                "Current regimen inadequate. Reports significant anxiety and insomnia.\n\n"
                "PMH: Chronic lumbar radiculopathy (L4-L5 disc herniation), "
                "Generalized Anxiety Disorder, Insomnia\n"
                "Allergies: Codeine (nausea)\n\n"
                "Objective:\n"
                "  Vitals: BP 130/82, HR 70, Temp 37.0°C\n"
                "  Neuro: Decreased sensation L5 dermatome left, "
                "weakness in left EHL 4/5\n"
                "  Straight leg raise: Positive at 40° left\n\n"
                "Assessment: Chronic lumbar radiculopathy with neuropathic pain\n\n"
                "Plan: Continue OxyContin 20mg BID. "
                "Add Xanax 0.5mg TID for anxiety. "
                "Gabapentin 300mg TID for neuropathic pain. "
                "MRI lumbar spine ordered. Neurosurgery referral."
            ),
            "labs": (
                "CBC: WNL\n"
                "BMP: WNL\n"
                "Urine Drug Screen: Positive for oxycodone (expected)\n"
                "LFTs: AST 28, ALT 32 (normal)"
            ),
            "meds": (
                "OxyContin 20mg PO BID\n"
                "Xanax 0.5mg PO TID\n"
                "Gabapentin 300mg PO TID\n"
                "Ibuprofen 400mg PO PRN\n"
                "Trazodone 50mg PO QHS PRN"
            ),
        },
    },
    {
        "name": "Ama Osei",
        "dob": "1962-06-25",
        "mrn": "MRN-4401",
        "encounter": {
            "note": (
                "Patient: Ama Osei\n"
                "DOB: 06/25/1962 | MRN: MRN-4401\n\n"
                "Subjective: 64F kidney transplant recipient (2019) presents "
                "with oral thrush and vaginal candidiasis. "
                "Reports white oral patches x5 days, vaginal itching.\n\n"
                "PMH: Renal transplant (living donor 2019), Hyperlipidemia, "
                "Type 2 Diabetes, Hypertension\n"
                "Allergies: None known\n\n"
                "Objective:\n"
                "  Vitals: BP 132/78, HR 72, Temp 37.1°C\n"
                "  Oral: White plaques on buccal mucosa, palate — "
                "scrapes off to reveal erythematous base\n"
                "  Transplant graft: Non-tender, no bruit changes\n\n"
                "Assessment: Oropharyngeal and vulvovaginal candidiasis "
                "in immunosuppressed transplant recipient\n\n"
                "Plan: Start Diflucan 200mg PO Day 1, then 100mg daily x 14 days. "
                "Continue current immunosuppression and statin. "
                "Recheck LFTs and tacrolimus level in 1 week."
            ),
            "labs": (
                "Tacrolimus level: 8.2 ng/mL (therapeutic)\n"
                "Creatinine: 1.4 mg/dL (stable)\n"
                "HbA1c: 7.8%\n"
                "LDL: 118 mg/dL\n"
                "LFTs: AST 22, ALT 28 (normal)"
            ),
            "meds": (
                "Tacrolimus 3mg PO BID\n"
                "Mycophenolate 500mg PO BID\n"
                "Prednisone 5mg PO Daily\n"
                "Lipitor 40mg PO Daily\n"
                "Diflucan 200mg PO (NEW — loading dose)\n"
                "Metformin 1000mg PO BID\n"
                "Amlodipine 10mg PO Daily"
            ),
        },
    },
]


def seed():
    ps = PatientService()
    created = 0
    skipped = 0

    for demo in DEMO_PATIENTS:
        patient = ps.create_patient(demo["name"], demo["dob"], mrn=demo.get("mrn", ""))

        if patient is None:
            print(f"  ⏭️  Skipped (already exists): {demo['name']}")
            skipped += 1
            continue

        # Save encounter
        enc = demo["encounter"]
        ps.save_encounter(
            patient_id=patient["id"],
            input_data=enc,
            report_data={},  # No audit report yet
        )

        print(f"  ✅ Created: {demo['name']} (ID: {patient['id']})")
        created += 1

    print(f"\nDone! Created {created}, skipped {skipped} (duplicates).")
    print(f"Total patients now: {len(ps.get_all_patients())}")


if __name__ == "__main__":
    print("🌱 Seeding SentinelMD with demo patients...\n")
    seed()
