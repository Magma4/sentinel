import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional

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
