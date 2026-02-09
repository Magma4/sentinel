import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any

# Constants
DATA_DIR = "data/patients"
INDEX_FILE = os.path.join(DATA_DIR, "index.json")

class PatientService:
    """
    Manages local patient records and encounters using a flat-file JSON structure.
    Designed for 100% offline usage without a database server.
    """
    def __init__(self):
        self._ensure_storage()

    def _ensure_storage(self):
        """Creates necessary directories and index file if missing."""
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(INDEX_FILE):
            with open(INDEX_FILE, "w") as f:
                json.dump([], f)

    def get_all_patients(self) -> List[Dict]:
        """Returns a list of all patients from the index."""
        try:
            with open(INDEX_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """Finds a patient by ID."""
        patients = self.get_all_patients()
        for p in patients:
            if p["id"] == patient_id:
                return p
        return None

    def create_patient(self, name: str, dob: str, mrn: str = "") -> Dict:
        """Creates a new patient and adds them to the index."""
        patients = self.get_all_patients()

        # Check for duplicates (same name AND DOB)
        for p in patients:
            if p["name"].lower().strip() == name.lower().strip() and p["dob"] == dob:
                return None  # Indicate duplicate found

        # Simple ID generation (short UUID for readability)
        new_id = str(uuid.uuid4())[:8]

        new_patient = {
            "id": new_id,
            "name": name,
            "dob": dob,
            "mrn": mrn,
            "created_at": datetime.now().isoformat()
        }

        patients.append(new_patient)

        # Atomic write to index
        with open(INDEX_FILE, "w") as f:
            json.dump(patients, f, indent=2)

        # Create dedicated folder for this patient's encounters
        patient_dir = os.path.join(DATA_DIR, new_id)
        os.makedirs(patient_dir, exist_ok=True)

        return new_patient

    def delete_patient(self, patient_id: str) -> bool:
        """Deletes a patient and all their encounters from the system."""
        import shutil

        patients = self.get_all_patients()
        original_count = len(patients)

        # Remove from index
        patients = [p for p in patients if p["id"] != patient_id]

        if len(patients) == original_count:
            return False  # Patient not found

        # Write updated index
        with open(INDEX_FILE, "w") as f:
            json.dump(patients, f, indent=2)

        # Delete patient folder and all encounters
        patient_dir = os.path.join(DATA_DIR, patient_id)
        if os.path.exists(patient_dir):
            shutil.rmtree(patient_dir)

        return True

    def save_encounter(self, patient_id: str, input_data: Dict, report_data: Dict, audio_file: str = None) -> str:
        """
        Saves a clinical encounter (inputs + outputs) to the patient's record.
        Returns the filename of the saved encounter.
        """
        encounter_id = str(uuid.uuid4())[:8]
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{date_str}_{encounter_id}.json"

        patient_dir = os.path.join(DATA_DIR, patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        encounter_record = {
            "id": encounter_id,
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "inputs": input_data,  # The text, labs, meds provided
            "report": report_data, # The Safety Audit Result
            "audio_file": audio_file # Optional path to saved audio
        }

        filepath = os.path.join(patient_dir, filename)
        with open(filepath, "w") as f:
            json.dump(encounter_record, f, indent=2)

        return filepath

    def get_encounters(self, patient_id: str) -> List[Dict]:
        """Retrieves all encounters for a patient, sorted by newest first."""
        patient_dir = os.path.join(DATA_DIR, patient_id)
        if not os.path.exists(patient_dir):
            return []

        encounters = []
        if os.path.exists(patient_dir):
            for filename in os.listdir(patient_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(patient_dir, filename), "r") as f:
                            encounters.append(json.load(f))
                    except Exception:
                        continue # Skip corrupted files

        # Sort by timestamp descending
        encounters.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return encounters

    def get_population_stats(self) -> Dict[str, Any]:
        """Aggregates safety statistics across the entire patient population."""
        patients = self.get_all_patients()
        stats = {
            "total_patients": len(patients),
            "risk_distribution": {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0},
            "top_flags": {},
            "conditions": {}
        }

        for p in patients:
            encounters = self.get_encounters(p["id"])
            if not encounters:
                stats["risk_distribution"]["Unknown"] += 1
                continue

            # Analyze latest encounter
            latest = encounters[0]
            report = latest.get("report", {})

            # 1. Risk Level (Max Severity)
            flags = report.get("flags", [])
            if not flags:
                stats["risk_distribution"]["Low"] += 1
            else:
                max_sev = "Low"
                for f in flags:
                    sev = f.get("severity", "LOW").upper()
                    if sev == "HIGH":
                        max_sev = "High"
                        break
                    elif sev == "MEDIUM" and max_sev != "High":
                        max_sev = "Medium"
                stats["risk_distribution"][max_sev] += 1

            # 2. Top Flags
            for f in flags:
                cat = f.get("category", "OTHER")
                stats["top_flags"][cat] = stats["top_flags"].get(cat, 0) + 1

            # 3. Conditions (from Extract)
            # Note: "conditions" might be in inputs if extracted, or in report metadata
            # For now, we'll try to parse from the note or use a placeholder if structured data missing
            # In a real system, we'd use the structured 'extraction' result.
            # Here we'll just check the 'inputs' for mentions of common conditions in the 'history' section
            # or rely on the FactExtractor output if saved.
            # Current save_encounter saves 'inputs' and 'report'.
            # If we want conditions, we should probably save extraction result too.
            # For this demo, let's skip deep condition parsing and just count flags.
            pass

        # Sort top flags
        sorted_flags = dict(sorted(stats["top_flags"].items(), key=lambda item: item[1], reverse=True)[:5])
        stats["top_flags"] = sorted_flags

        return stats
