import json
import os
from typing import Dict, Any, Optional
from .llm_client import LocalLLMClient
from .preprocess import trim_note, trim_labs, trim_meds

class FactExtractor:
    """Extracts clinical facts using MedGemma or Mock backend."""

    description = "Extracts clinical facts using MedGemma or Mock backend."

    def __init__(self, backend_url: Optional[str] = None, backend_type: str = "mock", model: str = "amsaravi/medgemma-4b-it:q6"):
        self.backend_url = backend_url
        self.backend_type = backend_type
        self.prompt_template = self._load_prompt()

        self.client = LocalLLMClient(
            backend=backend_type,
            model=model,
            host=backend_url
        )

    def _load_prompt(self) -> str:
        prompt_path = os.path.join("prompts", "extract_facts.md")
        try:
            with open(prompt_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "Error: Prompt file not found."

    def extract_facts(self, note: str, labs: str, meds: str, llm_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Orchestrates fact extraction from clinical text inputs."""
        import time
        start_time = time.time()

        if self.backend_type == "mock":
             return self._mock_extraction(note, labs, meds)

        # Trimming Context
        trimmed_note = trim_note(note)
        trimmed_labs = trim_labs(labs)
        trimmed_meds = trim_meds(meds)

        input_vars = {
            "NOTE_TEXT": trimmed_note,
            "LAB_TEXT": trimmed_labs,
            "MED_TEXT": trimmed_meds
        }

        try:
            if llm_options is None:
                llm_options = {
                    "num_ctx": 2048,
                    "num_predict": 280,
                    "temperature": 0.0
                }
            result = self.client.generate_json(self.prompt_template, input_vars, llm_options)


            # Shape Validation (Loose)
            required_keys = ["medications", "allergies", "conditions"]
            if not all(k in result for k in required_keys):
                 for k in required_keys:
                     if k not in result:
                         result[k] = []

            # Metadata
            duration = time.time() - start_time
            result["_metadata"] = {"execution_time": duration}

            return result

        except Exception as e:
            return {
                "medications": [],
                "allergies": [],
                "conditions": [],
                "_metadata": {"execution_time": time.time() - start_time},
                "error": str(e)
            }

    def parse_dictation(self, transcript: str) -> Dict[str, str]:
        """Parses a raw dictation transcript into structured sections."""
        if not transcript or not transcript.strip():
            return {"note_section": "", "medications": "", "labs": ""}

        if self.backend_type == "mock":
            # Simple heuristic mock
            lower = transcript.lower()
            meds = []
            labs = []
            note = transcript
            if "lisinopril" in lower: meds.append("Lisinopril 10mg")
            if "aspirin" in lower: meds.append("Aspirin 81mg")
            if "troponin" in lower: labs.append("Troponin: 0.04")
            if "creatinine" in lower: labs.append("Creatinine: 1.1")

            return {
                "note_section": note,
                "medications": "\n".join(meds),
                "labs": "\n".join(labs)
            }

        # Real LLM call
        prompt_path = os.path.join("prompts", "dictation_parse.md")
        try:
            with open(prompt_path, "r") as f:
                prompt_template = f.read()
        except FileNotFoundError:
            return {"note_section": transcript, "medications": "", "labs": "", "error": "Prompt missing"}

        try:
            input_vars = {"TRANSCRIPT": transcript}
            llm_options = {"num_ctx": 4096, "temperature": 0.0}
            result = self.client.generate_json(prompt_template, input_vars, llm_options)

            # Ensure keys exist
            return {
                "note_section": result.get("note_section", transcript),
                "medications": result.get("medications", ""),
                "labs": result.get("labs", "")
            }
        except Exception as e:
            return {"note_section": transcript, "medications": "", "labs": "", "error": str(e)}

    def _mock_extraction(self, note: str, labs: str, meds: str) -> Dict[str, Any]:
        """Deterministic regex-based mock for testing."""
        mock_data = {
            "patient_age": "UNKNOWN",
            "sex": "UNKNOWN",
            "key_conditions": [],
            "current_meds": [],
            "allergies": [],
            "labs": [],
            "clinician_assertions": [],
            # Standard keys
            "medications": [],
            "conditions": []
        }

        text_lower = (note + " " + labs + " " + meds).lower()

        # 1. Allergies
        if "penicillin" in text_lower and ("allergy" in text_lower or "hives" in text_lower):
            mock_data["allergies"].append("Penicillin")

        # 2. Meds
        known_meds = ["amoxicillin", "metformin", "lisinopril", "atorvastatin", "spironolactone"]
        for med in known_meds:
            if med in text_lower:
                mock_data["medications"].append(med.capitalize())
                mock_data["current_meds"].append(med.capitalize())

        # 3. Conditions
        if "diabetes" in text_lower:
            mock_data["conditions"].append("Diabetes Mellitus")
            mock_data["key_conditions"].append("Diabetes Mellitus")
        if "hypertension" in text_lower:
             mock_data["conditions"].append("Hypertension")
             mock_data["key_conditions"].append("Hypertension")

        mock_data["_metadata"] = {"execution_time": 0.001}
        return mock_data
