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
    """Core auditing engine using MedGemma or Rule-based Mock."""

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
        """Executes the full safety audit pipeline."""
        import time
        start_time = time.time()

        # 1. Trimming
        trimmed_note = trim_note(note)
        trimmed_labs = trim_labs(labs)
        trimmed_meds = trim_meds(meds)

        input_vars = {
            "extracted_facts": json.dumps(facts_json, indent=2),
            "note": trimmed_note,
            "labs": trimmed_labs,
            "meds": trimmed_meds
        }

        # 2. Inference
        raw_response = {}
        if self.backend_type == "mock":
            raw_response = self._mock_audit(note, labs, meds)
        else:
            # RAG-Lite: Fetch guidelines
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
                    summary=f"LLM Error: {str(e)}",
                    flags=[],
                    metadata={"error": str(e)}
                )

        # 3. Post-Processing
        try:
            if "flags" not in raw_response:
                raw_response["flags"] = []

            # 3.1 Repair Evidence
            self._repair_evidence(raw_response["flags"], note, labs, meds)

            # Metadata
            duration = time.time() - start_time
            self.model_metadata["audit_runtime"] = duration
            self.model_metadata["analysis_trace"] = {
                "step_1": raw_response.get("analysis_step_1_allergies", []),
                "step_2": raw_response.get("analysis_step_2_meds", []),
                "step_3": raw_response.get("analysis_step_3_conflicts", "No analysis provided.")
            }

            if "_metadata" in facts_json:
                self.model_metadata["extract_runtime"] = facts_json["_metadata"].get("execution_time", 0)

            report = SafetyReport(
                patient_id=raw_response.get("patient_id", "UNKNOWN"),
                summary=raw_response.get("summary", "Analysis complete."),
                flags=raw_response.get("flags", []),
                missing_info_questions=raw_response.get("missing_info_questions", []),
                metadata=self.model_metadata
            )

            # 3.2 Guardrails & Calibration
            report = validate_report_guardrails(report)
            report = calibrate_confidence(report)

            # 3.3 Safety Gating
            report = gate_safety_flags(report)

            return report

        except ValueError as e:
             # Guardrail Block
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
        """Retrieves keyword-matched safety guidelines from local knowledge base."""
        try:
            with open("data/guidelines/general_safety.txt", "r") as f:
                content = f.read()

            blocks = content.split("\n\n")
            relevant = []
            combined_text = (note + " " + labs + " " + meds).lower()

            for block in blocks:
                # Soft RAG Heuristics
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
        """Standardizes evidence structure and attempts to verify source via Fuzzy Matching."""
        import difflib

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
                    # Fuzzy Verification
                    # 1. Try Exact First
                    verified_source, verified_quote = self._find_best_source(quote, note, labs, meds)

                    if verified_source != "UNKNOWN":
                        source = verified_source
                        quote = verified_quote # Correct the quote to match the text exactly
                    else:
                         # 2. Try Fuzzy (Difflib)
                         # Search distinct 100-char chunks for matches? No, simplify:
                         # Scan combined text?
                         best_match, best_src, score = self._fuzzy_search(quote, note, labs, meds)
                         if score > 0.85: # Strict threshold
                             source = best_src
                             quote = best_match

                # Remove hallucinated evidence if still UNKNOWN?
                # Policy: Keep it but mark as UNKNOWN so Gating drops it.

                if quote:
                     # Auto-Highlight Numbers
                     import re
                     highlighted = quote
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

    def _find_best_source(self, quote: str, note: str, labs: str, meds: str):
        q_lower = quote.lower().strip()
        if not q_lower: return "UNKNOWN", quote

        if q_lower in note.lower(): return "NOTE", quote
        if q_lower in labs.lower(): return "LABS", quote
        if q_lower in meds.lower(): return "MEDS", quote
        return "UNKNOWN", quote

    def _fuzzy_search(self, quote: str, note: str, labs: str, meds: str):
        import difflib

        candidates = []
        # Create sliding window candidates? Too slow.
        # Use simple get_close_matches on lines

        all_lines = []
        for line in note.split('\n'): all_lines.append((line, "NOTE"))
        for line in labs.split('\n'): all_lines.append((line, "LABS"))
        for line in meds.split('\n'): all_lines.append((line, "MEDS"))

        best_ratio = 0.0
        best_match = quote
        best_src = "UNKNOWN"

        # Heuristic: Check against lines
        # This is rough but fast. Ideally we'd scan n-grams.
        # Check against sentences

        for line, src in all_lines:
            if not line.strip(): continue
            ratio = difflib.SequenceMatcher(None, quote.lower(), line.lower()).ratio()
            # If line is huge and quote is small, ratio drops. partial ratio needed.
            if quote.lower() in line.lower():
                return line, src, 1.0 # Substring match

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = line # Replace hallucination with real line
                best_src = src

        return best_match, best_src, best_ratio


    def _mock_audit(self, note: str, labs: str, meds: str) -> Dict[str, Any]:
        """Offline Rule-based Mock for demo scenarios."""
        flags = []
        from .evidence import find_verbatim_quote, build_evidence

        # Rule 1: Penicillin (Scenario 3)
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
