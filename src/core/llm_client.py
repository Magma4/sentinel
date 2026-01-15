import requests
import json
import re
from typing import Dict, Any, Optional


def repair_json(raw_json: str) -> str:
    """
    Repairs common JSON formatting errors from LLM outputs.
    - Fixes missing colons: "key","value" -> "key": "value"
    - Removes trailing commas: {...,} -> {...}
    - Fixes missing commas between properties
    """
    if not raw_json:
        return raw_json

    # Fix: "property_name","value" -> "property_name": "value"
    # This handles the specific error seen: "analysis_step_1_allergies","UNKNOWN"
    repaired = re.sub(r'"([^"]+)"\s*,\s*"([^"]+)"(\s*[,}])', r'"\1": "\2"\3', raw_json)

    # Fix: "property_name",[...] -> "property_name": [...]
    repaired = re.sub(r'"([^"]+)"\s*,\s*(\[)', r'"\1": \2', repaired)

    # Remove trailing commas before closing braces/brackets
    repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)

    return repaired


class LocalLLMClient:
    """
    A simple client for interacting with local LLMs (Ollama) or a Mock backend.
    Enforces JSON outputs.
    """
    def __init__(self, backend: str, model: str, host: Optional[str] = None):
        """
        Initialize the client.
        :param backend: "mock" or "ollama"
        :param model: Model name (e.g. "medgemma")
        :param host: URL for Ollama (default: http://localhost:11434)
        """
        self.backend = backend.lower()
        self.model = model
        self.host = host or "http://localhost:11434"

        if self.backend not in ["mock", "ollama"]:
            raise ValueError(f"Unsupported backend: {backend}. Use 'mock' or 'ollama'.")

    def generate_json(self, prompt: str, input_vars: Dict[str, Any], llm_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generates JSON output from the LLM.
        :param prompt: The prompt template string
        :param input_vars: Dictionary of variables to format into the prompt
        :param llm_options: Optional dict of Ollama options to override defaults
        :return: Parsed JSON dictionary
        """
        # 1. format prompt
        try:
            formatted_prompt = prompt.format(**input_vars)
        except KeyError as e:
            raise ValueError(f"Missing input variable for prompt: {e}")

        # 2. Call backend
        if self.backend == "mock":
            return self._call_mock(formatted_prompt)
        elif self.backend == "ollama":
            return self._call_ollama(formatted_prompt, llm_options)

        return {}

    def _call_mock(self, prompt: str) -> Dict[str, Any]:
        """Returns deterministic canned outputs for testing."""
        # Simple canned response structure
        return {
            "mock_response": True,
            "note": "This is a mock response.",
            "flags": []
        }

    def _call_ollama(self, prompt: str, llm_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calls Ollama API with format='json' and retries on failure."""
        url = f"{self.host}/api/generate"

        # Speed-focused defaults
        default_options = {
            "temperature": 0.0,
            "num_ctx": 2048,
            "num_predict": 420,
            "top_k": 40,
            "top_p": 0.9
        }

        # Merge with user-provided options
        options = {**default_options, **(llm_options or {})}

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": options,
            "stop": ["```", "<start_of_turn>"]
        }

        max_retries = 2
        last_error = None
        last_raw_response = None

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()

                data = response.json()
                raw_json = data.get("response", "{}")
                last_raw_response = raw_json

                # Repair common JSON formatting errors before parsing
                repaired_json = repair_json(raw_json)

                # Parse the inner JSON string returned by the LLM
                return json.loads(repaired_json)

            except json.JSONDecodeError as e:
                last_error = e
                continue
            except requests.exceptions.RequestException as e:
                # Check for 404 (Model not found)
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                    raise RuntimeError(
                        f"Model '{self.model}' not found on Ollama server. "
                        f"Run `ollama pull {self.model}` in your terminal."
                    )
                raise RuntimeError(f"Ollama connection failed: {str(e)}")

        # If we exhausted retries, provide detailed error with response preview
        preview = last_raw_response[:300] if last_raw_response else "No response captured"
        raise RuntimeError(
            f"Failed to parse valid JSON after {max_retries} attempts.\n"
            f"Last error: {str(last_error)}\n"
            f"Response preview: {preview}..."
        )
