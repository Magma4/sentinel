import requests
import json
import re
from typing import Dict, Any, Optional, List


def repair_json(raw_json: str) -> str:
    """Repairs malformed JSON common in LLM outputs (missing colons, trailing commas)."""
    if not raw_json:
        return raw_json

    # "key","value" -> "key": "value"
    repaired = re.sub(r'"([^"]+)"\s*,\s*"([^"]+)"(\s*[,}])', r'"\1": "\2"\3', raw_json)

    # "key",[...] -> "key": [...]
    repaired = re.sub(r'"([^"]+)"\s*,\s*(\[)', r'"\1": \2', repaired)

    # Remove trailing commas
    repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)

    return repaired


class LocalLLMClient:
    """Interacts with local LLMs (Ollama) or Mock backend, enforcing strict JSON output."""

    def __init__(self, backend: str, model: str, host: Optional[str] = None):
        import os
        self.backend = backend.lower()
        self.model = model
        # Env > Arg > Default
        self.host = os.getenv("OLLAMA_HOST") or host or "http://localhost:11434"

        if self.backend not in ["mock", "ollama"]:
            raise ValueError(f"Unsupported backend: {backend}. Use 'mock' or 'ollama'.")

    def generate_json(self, prompt: str, input_vars: Dict[str, Any], llm_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generates validated JSON from the LLM based on input variables."""
        # 1. Format
        try:
            formatted_prompt = prompt.format(**input_vars)
        except KeyError as e:
            raise ValueError(f"Missing input variable for prompt: {e}")

        # 2. Execute
        if self.backend == "mock":
            return self._call_mock(formatted_prompt)
        elif self.backend == "ollama":
            return self._call_ollama(formatted_prompt, llm_options)

        return {}

    def generate_text(self, prompt: str, llm_options: Optional[Dict[str, Any]] = None) -> str:
        """Generates raw text output from the LLM."""
        if self.backend == "mock":
             return "This is a mock chat response."
        elif self.backend == "ollama":
             return self._call_ollama(prompt, llm_options, output_format="text")
        return ""

    def _call_mock(self, prompt: str) -> Dict[str, Any]:
        """Returns deterministic canned outputs for testing."""
        return {
            "mock_response": True,
            "note": "This is a mock response.",
            "flags": []
        }

    def _call_ollama(self, prompt: str, llm_options: Optional[Dict[str, Any]] = None, output_format: str = "json") -> Any:
        """Calls Ollama API with retries, JSON repair, and prompt caching via keep_alive."""
        url = f"{self.host}/api/generate"

        # Defaults optimizing for speed
        default_options = {
            "temperature": 0.0,
            "num_ctx": 2048,
            "num_predict": 420,
            "top_k": 40,
            "top_p": 0.9
        }

        options = {**default_options, **(llm_options or {})}

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
            "stop": ["```", "<start_of_turn>"],
            # PROMPT CACHING: Keep model loaded in memory for 10 minutes
            # This enables KV cache reuse for repeated prompts with same prefix
            "keep_alive": "10m"
        }

        if output_format == "json":
            payload["format"] = "json"

        max_retries = 2
        last_error = None
        last_raw_response = None

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()

                data = response.json()
                raw_response = data.get("response", "")
                last_raw_response = raw_response

                if output_format == "text":
                    return raw_response

                # Parse JSON
                repaired_json = repair_json(raw_response)
                return json.loads(repaired_json)

            except json.JSONDecodeError as e:
                last_error = e
                continue
            except requests.exceptions.RequestException as e:
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                    raise RuntimeError(
                        f"Model '{self.model}' not found. Run `ollama pull {self.model}`."
                    )
                raise RuntimeError(f"Ollama connection failed: {str(e)}")

        # Fallbacks
        if output_format == "text":
             return last_raw_response or "Error: No response."

        preview = last_raw_response[:300] if last_raw_response else "No response captured"
        raise RuntimeError(
            f"JSON parsing failed after {max_retries} attempts.\n"
            f"Error: {str(last_error)}\n"
            f"Preview: {preview}..."
        )
