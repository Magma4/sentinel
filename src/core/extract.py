import json
import os
from typing import Dict, Any, Optional
from .llm_client import LocalLLMClient
from .preprocess import trim_note, trim_labs, trim_meds

class FactExtractor:
    """
    Extracts clinical facts from patient records using MedGemma (via LocalLLM) or Mock.
    """

    def __init__(self, backend_url: Optional[str] = None, backend_type: str = "mock"):
        """
        :param backend_url: URL for Ollama (optional)
        :param backend_type: "mock" or "ollama"
        """
        self.backend_url = backend_url
        self.backend_type = backend_type
        self.prompt_template = self._load_prompt()

        # Initialize Client
        # If mock, model name is arbitrary.
        self.client = LocalLLMClient(
            backend=backend_type,
            model="medgemma",
            host=backend_url
        )

    def _load_prompt(self) -> str:
        # Load from the requested path
        prompt_path = os.path.join("prompts", "extract_facts.md")
        try:
            with open(prompt_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "Error: Prompt file not found."

    def extract_facts(self, note: str, labs: str, meds: str, llm_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extracts facts from clinical text parts.
        """
        import time
        start_time = time.time()

        # If backend is mock, use our specialized robust regex mock
        # instead of the generic LLMClient mock which might be too simple.
        if self.backend_type == "mock":
             return self._mock_extraction(note, labs, meds)

        # Prepare inputs for the prompt
        # The prompt uses {{NOTE_TEXT}}, {{LAB_TEXT}}, {{MED_TEXT}} (checking prompt file in next step to be sure,
        # but user request said "Format input variables {NOTE_TEXT, LAB_TEXT, MED_TEXT}")
        # Note: Previous file used {{input_text}} as a single block. I should align with USER REQUEST.
        # "Format input variables {NOTE_TEXT, LAB_TEXT, MED_TEXT}"

        # NOTE: self.client.generate_json expects a prompt string and input_vars dict.
        # But wait, generate_json takes (prompt: str, input_vars: dict)
        # So I shouldn't replace it here, I should pass input_vars to client.

        # Apply trimming for model context (original text preserved for evidence)
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


            # Validate Basic Shape
            required_keys = ["medications", "allergies", "conditions"]
            # Flexible validation: at least some keys present?
            # Or just return raw result and let Pydantic strictness downstream handle it?
            # User request: "Validate output shape (basic required keys exist)"
            if not all(k in result for k in required_keys):
                 # Fallback or partial?
                 # Let's ensure they exist
                 for k in required_keys:
                     if k not in result:
                         result[k] = []

            # Add Runtime Metadata
            duration = time.time() - start_time
            result["_metadata"] = {"execution_time": duration}

            return result

        except Exception as e:

            # Fallback to empty structure to prevent app crash
            return {
                "medications": [],
                "allergies": [],
                "conditions": [],
                "_metadata": {"execution_time": time.time() - start_time},
                "error": str(e)
            }

    def _mock_extraction(self, note: str, labs: str, meds: str) -> Dict[str, Any]:
        """
        Deterministic mock extraction based on simple keyword finding.
        Returns the structure expected by the audit step.
        """
        mock_data = {
            "patient_age": "UNKNOWN",
            "sex": "UNKNOWN",
            "key_conditions": [],
            "current_meds": [],
            "allergies": [],
            "labs": [],
            "clinician_assertions": [],
            # Standard keys for downstream compatibility
            "medications": [],
            "conditions": []
        }

        text_lower = (note + " " + labs + " " + meds).lower()

        # 1. Allergies
        if "penicillin" in text_lower and ("allergy" in text_lower or "hives" in text_lower):
            mock_data["allergies"].append("Penicillin")

        # 2. Meds (Simple keyword extraction)
        known_meds = ["amoxicillin", "metformin", "lisinopril", "atorvastatin", "spironolactone"]
        for med in known_meds:
            if med in text_lower:
                mock_data["medications"].append(med.capitalize())
                mock_data["current_meds"].append(med.capitalize())

        # 3. Conditions (Simple keyword extraction)
        if "diabetes" in text_lower:
            mock_data["conditions"].append("Diabetes Mellitus")
            mock_data["key_conditions"].append("Diabetes Mellitus")
        if "hypertension" in text_lower:
             mock_data["conditions"].append("Hypertension")
             mock_data["key_conditions"].append("Hypertension")

        mock_data["_metadata"] = {"execution_time": 0.001}
        return mock_data
