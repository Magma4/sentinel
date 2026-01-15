import json
import os
import requests
import re
from typing import Optional, Dict, Any, List
from .schema import SafetyReport, SafetyFlag, SafetySeverity, SafetyCategory, Evidence

from .llm_client import LocalLLMClient
from .guardrails import validate_report_guardrails
from .gating import gate_safety_flags, calibrate_confidence
from .preprocess import trim_note, trim_labs, trim_meds

class SafetyAuditor:
    """
    Audits patient records using MedGemma (via Local LLM) or a rule-based Mock.
    """

    def __init__(self, backend_url: Optional[str] = None, backend_type: str = "mock"):
        self.backend_url = backend_url
        self.backend_type = backend_type
        self.prompt_template = self._load_prompt()
        self.model_metadata = {
            "name": "MedGemma-2b-v1" if backend_type == "ollama" else "SentinelMD-Mock-Rules",
            "quantization": "q4_k_m" if backend_type == "ollama" else "N/A",
            "runtime": "Local Ollama" if backend_type == "ollama" else "Python Native"
        }

        self.client = LocalLLMClient(
            backend=backend_type,
            model="medgemma",
            host=backend_url
        )

    def _load_prompt(self) -> str:
        try:
            with open("prompts/safety_audit.md", "r") as f:
                return f.read()
        except FileNotFoundError:
            return "Error: Prompt file not found."

    def run_audit(self, facts_json: Dict, note: str, labs: str, meds: str, llm_options: Optional[Dict[str, Any]] = None) -> SafetyReport:
        """
        """
        import time
        start_time = time.time()

        # 1. Prepare Inputs (apply trimming for model context)
        trimmed_note = trim_note(note)
        trimmed_labs = trim_labs(labs)
        trimmed_meds = trim_meds(meds)

        input_vars = {
            "extracted_facts": json.dumps(facts_json, indent=2),
            "note": trimmed_note,
            "labs": trimmed_labs,
            "meds": trimmed_meds
        }

        # 2. Call Backend (original text used for evidence inference)
        raw_response = {}
        if self.backend_type == "mock":
            raw_response = self._mock_audit(note, labs, meds)
        else:
            # RAG-Lite: Fetch relevant guidelines (use original for keyword matching)
            guidelines = self._get_relevant_guidelines(note, labs, meds)

            input_vars["guidelines"] = guidelines

            try:
                if llm_options is None:
                    llm_options = {
                        "num_ctx": 2048,
                        "num_predict": 420,
                        "temperature": 0.0
                    }
                raw_response = self.client.generate_json(self.prompt_template, input_vars, llm_options)
            except Exception as e:
                return SafetyReport(
                    patient_id="ERROR",
                    summary=f"LLM Generation Error: {str(e)}",
                    flags=[],
                    metadata={"error": str(e)}
                )

        # 3. Parse & Validate
        try:
            if "flags" not in raw_response:
                raw_response["flags"] = []

            # 3.1 Repair Evidence Format (Upcast Strings to Objects)
            self._repair_evidence(raw_response["flags"], note, labs, meds)

            # Create preliminary report
            # Capture total time
            duration = time.time() - start_time
            self.model_metadata["audit_runtime"] = duration

            # Capture Chain of Thought / Analysis Steps
            self.model_metadata["analysis_trace"] = {
                "step_1": raw_response.get("analysis_step_1_allergies", []),
                "step_2": raw_response.get("analysis_step_2_meds", []),
                "step_3": raw_response.get("analysis_step_3_conflicts", "No analysis provided.")
            }

            # Incorporate extraction time if present
            if "_metadata" in facts_json:
                self.model_metadata["extract_runtime"] = facts_json["_metadata"].get("execution_time", 0)

            report = SafetyReport(
                patient_id=raw_response.get("patient_id", "UNKNOWN"),
                summary=raw_response.get("summary", "Analysis complete."),
                flags=raw_response.get("flags", []),
                missing_info_questions=raw_response.get("missing_info_questions", []),
                metadata=self.model_metadata
            )

            # 3.2 Guardrails: Validate the report
            # This raises ValueError if the report violates safety policies
            report = validate_report_guardrails(report)

            # 3.2.1 Confidence Calibration
            report = calibrate_confidence(report)

            # 3.3 Safety Gate: Filter low-quality flags
            report = gate_safety_flags(report)

            return report

        except ValueError as e:
             # Guardrail Violation: Return a "Safe" Error Report so UI shows the block
             err_meta = self.model_metadata.copy()
             err_meta["error"] = f"Guardrail Blocked: {str(e)}"
             return SafetyReport(
                patient_id="GUARDRAIL_BLOCKED",
                summary=f"Safety Guardrail Violation: {str(e)}",
                flags=[],
                metadata=err_meta
            )
        except Exception as e:
            return SafetyReport(
                patient_id="ERROR",
                summary=f"Validation Error: {str(e)}",
                flags=[],
                metadata={"error": str(e)}
            )

    def _get_relevant_guidelines(self, note: str, labs: str, meds: str) -> str:
        """
        Simple keyword-based retrieval of safety guidelines.
        """
        try:
            with open("data/guidelines/general_safety.txt", "r") as f:
                content = f.read()

            # Split into blocks
            blocks = content.split("\n\n")
            relevant = []

            combined_text = (note + " " + labs + " " + meds).lower()

            for block in blocks:
                # Simple heuristic: if any keyword from the header or body matches
                # Ideally, we map specific keywords to blocks.
                # For now, let's just dump all of them if the file is small (< 10 rules).
                # But to demonstrate RAG, let's do soft filtering:

                # If block mentions "Metformin" and text has "Metformin", keep it.
                # If block mentions "Potassium" and text has "Potassium" or "K", keep it.

                if "metformin" in block.lower() and "metformin" in combined_text:
                    relevant.append(block)
                elif "potassium" in block.lower() and ("potassium" in combined_text or " k " in combined_text):
                    relevant.append(block)
                elif "beta-blocker" in block.lower() and ("metoprolol" in combined_text or "atenolol" in combined_text):
                    relevant.append(block)
                elif "penicillin" in block.lower() and ("allergy" in combined_text):
                    relevant.append(block)

            if not relevant:
                return "No specific guidelines found for this context."

            return "\n\n".join(relevant)

        except Exception:
            return "Guidelines unavailable."

    def _repair_evidence(self, flags: List[Dict], note: str, labs: str, meds: str):
        """
        Fixes evidence format issues commonly caused by smaller LLMs.
        Converts ['Quote String'] -> [{'quote': 'Quote String', 'source': '...'}]
        """
        for f in flags:
            ev_list = f.get("evidence", [])
            if not isinstance(ev_list, list):
                f["evidence"] = []
                continue

            repaired_ev = []
            for item in ev_list:
                quote = ""
                source = "UNKNOWN"

                if isinstance(item, str):
                    quote = item
                elif isinstance(item, dict) and "quote" in item:
                    quote = item["quote"]
                    source = item.get("source", "UNKNOWN")

                if quote:
                     # Try to infer source if unknown
                     if source == "UNKNOWN":
                         source = self._infer_evidence_source(quote, note, labs, meds)

                     # Auto-highlight numbers if not present
                     import re
                     highlighted = quote
                     # Highlight style: Yellow background for high visibility
                     numeric_pattern = r'\b\d+(\.\d+)?(/ \d+)?\b'
                     def repl(match):
                         return f"<span style='background-color: #fff3cd; color: #856404; padding: 0 2px; border-radius: 2px;'>{match.group(0)}</span>"
                     highlighted = re.sub(numeric_pattern, repl, quote)

                     repaired_ev.append({
                         "quote": quote,
                         "highlighted_text": highlighted,
                         "source": source
                     })

            f["evidence"] = repaired_ev

    def _infer_evidence_source(self, quote: str, note: str, labs: str, meds: str) -> str:
        """
        Helper to guess where a quote came from (Note, Labs, or Meds).
        Uses token overlap for robustness against minor LLM modifications.
        """
        def get_best_source(q_tokens: set) -> str:
            # Score each source by overlap
            scores = {
                "NOTE": len(q_tokens.intersection(set(note.lower().split()))),
                "LABS": len(q_tokens.intersection(set(labs.lower().split()))),
                "MEDS": len(q_tokens.intersection(set(meds.lower().split())))
            }
            best = max(scores, key=scores.get)
            if scores[best] > 0: # At least one word match
                return best
            return "UNKNOWN"

        q_lower = quote.lower()

        # 1. Exact Match (Best)
        if q_lower in note.lower(): return "NOTE"
        if q_lower in labs.lower(): return "LABS"
        if q_lower in meds.lower(): return "MEDS"

        # 2. Token Overlap (Fallback)
        # Remove common stop words to avoid matching "is", "a", etc.
        stops = {"patient", "has", "is", "a", "the", "with", "allergy", "hives"}
        # Note: "allergy" is common but also a keyword. Let's be careful.
        # Actually "allergy" is often in the note summary or allergy list.
        tokens = set(q_lower.replace(":","").replace(".","").split()) - stops

        if tokens:
            return get_best_source(tokens)

        return "UNKNOWN"


    def _mock_audit(self, note: str, labs: str, meds: str) -> Dict[str, Any]:
        """
        Rule-based Mock for offline testing.
        Uses src.core.evidence helpers for strict grounding.
        """
        flags = []
        from .evidence import find_verbatim_quote, build_evidence

        # Rule 1: Penicillin (Scenario 3)
        # Check NOTE for "penicillin" AND ("allergy" OR "hives")
        # Check MEDS for "amoxicillin"
        pen_quote = find_verbatim_quote(note, ["penicillin"])
        amox_quote = find_verbatim_quote(meds, ["amoxicillin"])

        is_allergy = "allergy" in note.lower() or "hives" in note.lower()

        if pen_quote and amox_quote and is_allergy:
            flags.append({
                "category": "MED_LAB_CONFLICT",
                "severity": "HIGH",
                "confidence": 1.0,
                "evidence": [
                    build_evidence("NOTE", note, pen_quote).model_dump(),
                    build_evidence("MEDS", meds, amox_quote).model_dump()
                ],
                "explanation": "Patient has a documented Penicillin allergy but is prescribed Amoxicillin (a Penicillin-class antibiotic).",
                "recommendation": "Review allergy history and confirm medication choice."
            })

        # Rule 2: Hyperkalemia (Scenario 2)
        # Check LABS for "Potassium" and "6.1"
        uk_quote = find_verbatim_quote(labs, ["Potassium", "6.1"])
        if uk_quote:
             flags.append({
                "category": "MISSING_WORKFLOW_STEP",
                "severity": "HIGH",
                "confidence": 0.9,
                "evidence": [
                     build_evidence("LABS", labs, uk_quote).model_dump()
                ],
                "explanation": "Critical Hyperkalemia (6.1) detected without evidence of repeat testing or ECG.",
                "recommendation": "Review Potassium level and need for ECG."
            })

        # Rule 3: Metformin/Renal (Scenario 1)
        # Check LABS for "Creatinine" and "1.7"
        # Check MEDS for "Metformin"
        creat_quote = find_verbatim_quote(labs, ["Creatinine", "1.7"])
        met_quote = find_verbatim_quote(meds, ["Metformin"])

        if creat_quote and met_quote:
             flags.append({
                    "category": "MED_LAB_CONFLICT",
                    "severity": "HIGH",
                    "confidence": 0.95,
                    "evidence": [
                        build_evidence("LABS", labs, creat_quote).model_dump(),
                        build_evidence("MEDS", meds, met_quote).model_dump()
                    ],
                    "explanation": "Metformin use contraindicated or requires dose adjustment with impaired renal function (Creatinine 1.7).",
                    "recommendation": "Review Metformin indications in setting of renal function."
                })

        return {
            "patient_id": "MOCK-PATIENT",
            "summary": f"Audit complete. Identified {len(flags)} safety issue(s).",
            "flags": flags,
            "analysis_step_1_allergies": ["Penicillin" if "allergy" in note.lower() else "None"],
            "analysis_step_2_meds": ["Amoxicillin", "Metformin"] if "Amoxicillin" in meds else ["None"],
            "analysis_step_3_conflicts": "Simulated Chain of Thought: Check 1 (Allergies) -> Found Penicillin. Check 2 (Meds) -> Found Amoxicillin. Conclusion -> Conflict detected.",
            "missing_info_questions": [
                "Is the Penicillin allergy confirmed or historical?",
                "Is there a recent ECG available to verify Potassium impact?",
                "Was the patient asked about other OTC medications (e.g. NSAIDs)?"
            ]
        }
