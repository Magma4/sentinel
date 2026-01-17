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
            "stop": ["```", "<start_of_turn>"]
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
