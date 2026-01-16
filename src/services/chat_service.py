from typing import Dict, List, Any
import hashlib
from src.domain.models import ChatSession, ChatState, AuditContext, ChatMessage
import logging
from src.adapters.ollama_adapter import ReviewEngineAdapter

logger = logging.getLogger("sentinel.services.chat")

class ChatService:
    """Manages the Safety Review Assistant state."""

    def __init__(self, review_engine: ReviewEngineAdapter):
        self.engine = review_engine

    def reset_session(self, session: ChatSession, audit_data: Dict, input_summary: str) -> ChatSession:
        """Invalidates chat history if audit inputs change."""
        current_fp = hashlib.md5(str(audit_data).encode()).hexdigest()

        if session.audit_fingerprint != current_fp:
            session = ChatSession()
            session.audit_fingerprint = current_fp
            session.state = ChatState.AUDIT_READY

            # Populate Context
            flags = audit_data.get("flags", [])
            evidence_idx = {}
            for i, f in enumerate(flags):
                fid = f"flag_{i}"
                ev_quotes = [e.get("quote", "") for e in f.get("evidence", [])]
                evidence_idx[fid] = " | ".join(ev_quotes)

            session.context = AuditContext(
                audit_json=audit_data,
                flags=flags,
                missing_info=audit_data.get("missing_info_questions", []),
                inputs_summary=input_summary,
                evidence_index=evidence_idx
            )
        return session

    def classify_query(self, user_text: str) -> Dict[str, Any]:
        """Guardrail: Block diagnostic or treatment intent."""
        text = user_text.lower()

        # Blocked Keywords
        disallowed_keywords = [
            "diagnose", "diagnosis", "cancer", "tumor", "fracture", "broken",
            "treatment", "prescribe", "dosage", "dose", "medication", "medicine",
            "change", "switch", "stop", "start", "interpret", "scan", "x-ray", "image",
            "what do i have", "am i sick"
        ]

        if any(k in text for k in disallowed_keywords):
            # Allow explanation queries "Why was this flag..."
            if "why" in text or "explain" in text or "flag" in text:
                pass
            else:
                return {
                    "allowed": False,
                    "reason": "Request involves diagnosis, treatment advice, or image interpretation.",
                    "category": "disallowed"
                }

        return {"allowed": True, "reason": "Safe clarification", "category": "clarify"}

    def build_prompt(self, context: AuditContext, history: List[ChatMessage], user_text: str) -> str:
        """Constructs system instruction grounded in audit context."""
        flags_desc = []
        for i, f in enumerate(context.flags):
             sev = f.get('severity', 'UNKNOWN')
             cat = f.get('category', 'ISSUE')
             exp = f.get('explanation', '')
             evid = context.evidence_index.get(f'flag_{i}', '')[:150]
             flags_desc.append(f"- [{sev}] {cat}: {exp}. (Evidence: {evid}...)")

        flags_block = "\n".join(flags_desc)

        # Instruction Template
        instruction = f"""
You are the SentinelMD Safety Assistant.
Your precise role is to explain the Safety Review findings to a clinician.

## STRICT RULES:
1. NO DIAGNOSIS: Do not diagnose conditions or interpret medical images.
2. NO TREATMENT: Do not recommend medications, dosages, or treatment plans.
3. GROUNDED ONLY: You can ONLY discuss the Flags and Missing Information listed below.
4. KEYWORD: Use "Review" or "Consider" instead of "must" or "should".

## REVIEW CONTEXT:
Input Summary: {context.inputs_summary}

Safety Flags Found:
{flags_block}

Missing Info / Clarifications:
{", ".join(context.missing_info)}

User Question: {user_text}

Answer concisely (3-4 sentences max). Use bullet points if listing evidence.
"""
        return instruction.strip()

    def generate_reply(self, session: ChatSession, user_query: str) -> str:
        """Orchestrates RAG-like reply generation."""
        # 1. Classify
        classification = self.classify_query(user_query)
        if not classification["allowed"]:
            return "This assistant can clarify the audit results only. It cannot provide diagnosis, treatment advice, or image interpretation."

        # 2. Build
        instruction = self.build_prompt(session.context, session.history, user_query)

        # 3. Generate
        try:
             opts = {"temperature": 0.0, "num_predict": 256}
             raw_resp = self.engine.generate_text(instruction, opts)

             clean_resp = raw_resp.replace("Assistant:", "").strip()
             return clean_resp
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            return "Local review engine unavailable."
