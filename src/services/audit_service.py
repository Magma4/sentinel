from typing import Dict, Any, List, Optional
import time
import logging
from src.domain.models import AuditReport, SafetyFlag, SafetySeverity, SafetyCategory, EvidenceQuote
from src.adapters.ollama_adapter import ReviewEngineAdapter
from src.core.ddi_checker import scan_medications, DDInteraction

logger = logging.getLogger("sentinel.services.audit")

class AuditService:
    """Core Clinical Safety Review Service."""

    def __init__(self, review_engine: ReviewEngineAdapter):
        self.engine = review_engine
        self.instruction_path = "prompts/safety_audit.md"

    def _load_instruction(self) -> str:
        try:
            with open(self.instruction_path, "r") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load instruction template: {e}")
            return "Analyze this clinical text for safety risks."

    @staticmethod
    def run_ddi_scan(meds_text: str) -> List[DDInteraction]:
        """Run the deterministic DDI pre-scan on medication text."""
        return scan_medications(meds_text)

    @staticmethod
    def _ddi_to_flags(interactions: List[DDInteraction]) -> List[SafetyFlag]:
        """Convert DDI results into SafetyFlag objects."""
        flags = []
        for ddi in interactions:
            sev = SafetySeverity(ddi.severity) if ddi.severity in ("HIGH", "MEDIUM", "LOW") else SafetySeverity.MEDIUM
            flags.append(SafetyFlag(
                category=SafetyCategory.MEDICATION_INTERACTION,
                severity=sev,
                explanation=f"Interaction: {ddi.drug_a.title()} + {ddi.drug_b.title()} — {ddi.mechanism}",
                recommendation=ddi.recommendation,
                evidence=[
                    EvidenceQuote(source="MEDS", quote=f"{ddi.drug_a.title()} + {ddi.drug_b.title()} (DDI database)")
                ],
            ))
        return flags

    def run_safety_review(self,
                          note_text: str,
                          labs_text: str,
                          meds_text: str,
                          config: Optional[Dict] = None) -> Optional[AuditReport]:
        """Executes the safety review pipeline."""
        t_start = time.time()

        # 1. Load Prompts & Prepare Data
        instruction_template = self._load_instruction()

        full_instruction = f"""
{instruction_template}

Input Data:
Clinical Note:
{note_text[:15000]}

Medications:
{meds_text[:5000]}

Labs:
{labs_text[:5000]}
"""

        # 2. Review Engine Inference
        try:
            engine_output = self.engine.run_structured_review(full_instruction, config)
        except Exception as e:
            logger.error(f"Review Engine failed: {e}")
            return AuditReport(
                summary="Review Engine unavailable. Automated checks could not complete.",
                flags=[],
                missing_info_questions=[],
                confidence_score=0.0,
                metadata={"error": str(e)}
            )

        # 3. Response Validation & Object Mapping
        try:
            flags = []
            for f in engine_output.get("flags", []):
                # Map Severity
                sev_str = f.get("severity", "MEDIUM").upper()
                try:
                    sev_enum = SafetySeverity(sev_str)
                except:
                    sev_enum = SafetySeverity.MEDIUM

                # Map Category
                cat_str = f.get("category", "OTHER").upper()
                try:
                    cat_enum = SafetyCategory(cat_str)
                except:
                    # Heuristic Fallback
                    low_exp = f.get("explanation", "").lower()
                    if "allergy" in low_exp or "allergic" in low_exp:
                        cat_enum = SafetyCategory.ALLERGY
                    elif "interaction" in low_exp and "medication" in low_exp:
                        cat_enum = SafetyCategory.MEDICATION_INTERACTION
                    else:
                        cat_enum = SafetyCategory.OTHER

                # Map Evidence
                ev_objs = []
                for e in f.get("evidence", []):
                    if isinstance(e, str):
                        ev_objs.append(EvidenceQuote(source="NOTE", quote=e))
                    else:
                        ev_objs.append(EvidenceQuote(source=e.get("source", "NOTE"), quote=e.get("quote", "")))

                flags.append(SafetyFlag(
                    category=cat_enum,
                    severity=sev_enum,
                    explanation=f.get("explanation", ""),
                    recommendation=f.get("recommendation", f.get("check_steps", "")),
                    evidence=ev_objs
                ))

            # 4. DDI pre-scan flags (deterministic)
            ddi_hits = self.run_ddi_scan(meds_text)
            ddi_flags = self._ddi_to_flags(ddi_hits)

            report = AuditReport(
                summary=engine_output.get("summary", "No summary provided."),
                flags=ddi_flags + flags,  # DDI flags first, then LLM flags
                missing_info_questions=engine_output.get("missing_info_questions", []),
                patient_demographics=engine_output.get("patient_demographics", None),
                confidence_score=engine_output.get("confidence_score", 0.8),
                metadata={
                    "engine_duration": time.time() - t_start,
                    "model": self.engine.model,
                    "ddi_scan": {"interactions": len(ddi_hits), "meds_parsed": len(ddi_hits)}
                }
            )
            return report

        except Exception as e:
            logger.error(f"Domain Mapping Error: {e}")
            return None
    def execute_billing_analysis(self, note_text: str) -> Dict[str, Any]:
        """Runs the Revenue Retina (billing analysis) pipeline."""
        try:
            return self.engine.run_billing_analysis(note_text)
        except Exception as e:
            logger.error(f"Billing analysis failed: {e}")
            return {"error": str(e)}

    def get_patient_instructions(self, note_text: str, language: str = "English", safety_flags: list = None) -> Dict[str, Any]:
        """Runs the Patient Translator pipeline."""
        try:
            return self.engine.generate_patient_instructions(note_text, language, safety_flags)
        except Exception as e:
            logger.error(f"Patient instructions failed: {e}")
            return {"error": str(e)}
