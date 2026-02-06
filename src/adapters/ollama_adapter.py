import requests
import json
import re
from typing import Dict, Any, Optional, List
import logging

# Logger
logger = logging.getLogger("sentinel.adapters.review_engine")

def repair_json(json_str: str) -> str:
    """Heuristic repair for generated JSON format faults."""
    json_str = json_str.strip()
    # Remove markdown code blocks
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    return json_str.strip()


class ReviewEngineAdapter:
    """
    Adapter for Local Review Engine (Ollama).
    Handles structured assessment (JSON) and general inquiries (Text).
    """

    def __init__(self, backend_url: str = "http://localhost:11434", model: str = "amsaravi/medgemma-4b-it:q6"):
        import os
        # Prioritize Environment Variable > Init Arg > Default
        env_host = os.getenv("OLLAMA_HOST")
        self.host = env_host if env_host else backend_url
        self.model = model

        # Check if remote or local
        self.backend = "ollama" if "localhost" in self.host or "127.0.0.1" in self.host or env_host else "mock"

    def check_connection(self) -> bool:
        if self.backend == "mock": return True
        try:
            requests.get(f"{self.host}/api/tags", timeout=2)
            return True
        except:
            return False

    def list_models(self) -> List[str]:
        if self.backend == "mock": return ["mock-model"]
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return [m["name"] for m in models]
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
        return []

    def generate_text(self, instruction: str, engine_options: Optional[Dict[str, Any]] = None) -> str:
        """Generates plain text response (Assistant mode)."""
        if self.backend == "mock":
             return "This is a mock review engine response."

        return self._call_engine(instruction, engine_options, output_format="text")

    def run_billing_analysis(self, note_text: str) -> Dict[str, Any]:
        """Runs ICD-10 and CPT analysis to suggest billing levels."""
        if self.backend == "mock":
             return {
                 "icd10": [{"code": "I10", "desc": "Essential (primary) hypertension"}],
                 "cpt": {"code": "99214", "desc": "Office visit, moderate complexity", "level": 4},
                 "opportunity": {
                     "found": True,
                     "potential_code": "99215",
                     "revenue_impact": "+$40",
                     "missing_elements": ["Document interpretation of labs (e.g. creatinine trend)", "Note total time > 40 mins"]
                 }
             }

        prompt = f"""
You are a Medical Billing Expert (CPC Certified) reviewing a clinical note.
Analyze the note for ICD-10 diagnosis codes and CPT Evaluation & Management (E/M) levels.

Determine the Current CPT Level (99202-99205 or 99212-99215) and identify if the documentation supports a higher level with minimal additions.

OUTPUT JSON FORMAT:
{{
  "icd10": [{{"code": "string", "desc": "string"}}],
  "cpt": {{"code": "string", "desc": "string", "level": integer}},
  "opportunity": {{
    "found": boolean,
    "potential_code": "string (e.g. 99215)",
    "revenue_impact": "string (e.g. +$40)",
    "missing_elements": ["list of specific things to document to reach higher level"]
  }}
}}

CLINICAL NOTE:
{note_text}
"""
        return self._call_engine(prompt, {"num_predict": 1024}, output_format="json")

    def generate_patient_instructions(self, note_text: str, language: str = "English", safety_flags: list = None) -> Dict[str, Any]:
        """Generates patient-friendly instructions in specified language."""
        if self.backend == "mock":
             return {
                 "summary": "You visited today because of high blood pressure.",
                 "key_takeaways": ["Your heart health is stable", "Diet changes are needed"],
                 "medication_instructions": ["Take Lisinopril 10mg once a day with breakfast"],
                 "terminology_map": [{"term": "Hypertension", "simple": "High Blood Pressure"}]
             }

        lang_instruction = "CRITICAL: OUTPUT ALL TEXT VALUES IN SPANISH (Español). Do not output English." if "Spanish" in language else "Output in English."

        flags_context = ""
        if safety_flags and len(safety_flags) > 0:
            flags_context = (
                f"🚨 URGENT SAFETY ALERT 🚨\n"
                f"The following risks were detected: {str(safety_flags)}\n"
                f"INSTRUCTIONS:\n"
                f"1. SUMMARY: If the doctor planned to start a flagged drug, DELETE that part of the plan. Replace it with: 'We need to double-check [Drug Name] for safety before starting.' Do NOT say both.\n"
                f"2. MEDICATIONS: List flagged drugs ONLY as '⚠️ [Drug Name]: ON HOLD (Talk to Doctor)'. Do not duplicate.\n"
            )

        prompt = f"""
You are a fastidious, compassionate doctor writing a 'Take Home Summary' for your patient.
Your goal is to explain their visit, diagnosis, and plan in simple, 5th-grade reading level language.

{lang_instruction}

{flags_context}

TASK:
1. Summarize the visit {language}.
2. Provide a 'Medical Decoder' for complex terms (e.g. Hypertension -> High Blood Pressure).
3. List 3 key takeaways.
4. Provide simple medication instructions. **CRITICAL: CHECK SAFETY CONTEXT FIRST.**

OUTPUT JSON FORMAT:
{{
  "summary": "string (warm tone, {language})",
  "key_takeaways": ["string (in {language})", "string (in {language})"],
  "medication_instructions": ["string (in {language})", "string (in {language})"],
  "terminology_map": [{{"term": "Medical Jargon (Original)", "simple": "Simple Explanation ({language}) "}}]
}}

CLINICAL NOTE:
{note_text}
"""
        return self._call_engine(prompt, {"num_predict": 1024}, output_format="json")

    def run_structured_review(self, instruction: str, engine_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Runs safety assessment returning strict JSON."""
        if self.backend == "mock":
             return self._call_mock(instruction)

        return self._call_engine(instruction, engine_options, output_format="json")

    def _call_mock(self, instruction: str) -> Dict[str, Any]:
        return {
            "mock_response": True,
            "summary": "Mock Audit: No risks found in demo mode.",
            "flags": [],
            "missing_info_questions": []
        }

    def _call_engine(self, instruction: str, engine_options: Optional[Dict[str, Any]] = None, output_format: str = "json") -> Any:
        """Execute request against Ollama API."""
        url = f"{self.host}/api/generate"

        # Defaults optimized for clinical extraction
        default_opts = {
            "temperature": 0.0,
            "num_ctx": 4096,
            "num_predict": 512,
            "top_k": 40,
            "top_p": 0.9
        }

        options = {**default_opts, **(engine_options or {})}

        payload = {
            "model": self.model,
            "prompt": instruction,
            "stream": False,
            "options": options,
            "stop": ["```", "<start_of_turn>"],
            # PROMPT CACHING: Keep model loaded for 10 minutes
            # Speeds up repeated calls by reusing loaded model weights
            "keep_alive": "10m"
        }

        if output_format == "json":
            payload["format"] = "json"

        max_retries = 2
        last_error = None
        last_raw = None

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=90)
                response.raise_for_status()

                data = response.json()
                raw_response = data.get("response", "")
                last_raw = raw_response

                if output_format == "text":
                    return raw_response

                # Parse
                repaired = repair_json(raw_response)
                return json.loads(repaired)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON Parse Error (Attempt {attempt+1}): {e}")
                last_error = e
                continue
            except requests.exceptions.RequestException as e:
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                    raise RuntimeError(f"Review Engine Model '{self.model}' not found. Run `ollama pull {self.model}`.")
                raise RuntimeError(f"Review Engine Connection Failed: {str(e)}")

        # Fallback
        if output_format == "text":
             return last_raw or "Error: Review Engine returned no output."

        preview = last_raw[:300] if last_raw else "None"
        raise RuntimeError(
            f"Review Engine failed to produce valid structured data after {max_retries} attempts.\n"
            f"Preview: {preview}..."
        )
