from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import hashlib
import time

class ChatState(Enum):
    IDLE = auto()
    AUDIT_READY = auto()
    VALIDATE_QUERY = auto()
    REFUSE = auto()
    BUILD_CONTEXT = auto()
    ANSWERING = auto()
    RENDER_ANSWER = auto()
    ERROR = auto()

@dataclass
class ChatMessage:
    role: str # "user" or "assistant"
    content: str
    ts: float = field(default_factory=time.time)

@dataclass
class AuditContext:
    audit_json: Dict
    flags: List[Dict]
    missing_info: List[str]
    inputs_summary: str
    evidence_index: Dict[str, str] # flag_id -> quote

@dataclass
class ChatSession:
    state: ChatState = ChatState.IDLE
    history: List[ChatMessage] = field(default_factory=list)
    last_error: Optional[str] = None
    audit_fingerprint: str = ""
    context: Optional[AuditContext] = None

def fingerprint_audit(audit_json: Dict) -> str:
    """Hashes audit content to detect staleness."""
    return hashlib.md5(str(audit_json).encode()).hexdigest()

def reset_session_for_new_audit(session: ChatSession, fingerprint: str, audit_data: Dict, input_summary: str) -> ChatSession:
    """Clears history if the underlying audit context changes."""
    if session.audit_fingerprint != fingerprint:
        session = ChatSession()
        session.audit_fingerprint = fingerprint
        session.state = ChatState.AUDIT_READY

        # Build Context
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

def classify_query(user_text: str) -> Dict[str, Any]:
    """Safety Guardrail: Rejects diagnostic or treatment queries."""
    text = user_text.lower()

    # DISALLOWED INTENTS
    disallowed_keywords = [
        "diagnose", "diagnosis", "cancer", "tumor", "fracture", "broken",
        "treatment", "prescribe", "dosage", "dose", "medication", "medicine",
        "change", "switch", "stop", "start", "interpret", "scan", "x-ray", "image",
        "what do i have", "am i sick"
    ]

    if any(k in text for k in disallowed_keywords):
        # Exception: "Why was this flag raised?" might contain "medication"
        if "why" in text or "explain" in text or "flag" in text:
            pass # Likely safe
        else:
            return {
                "allowed": False,
                "reason": "Request involves diagnosis, treatment advice, or image interpretation.",
                "category": "disallowed"
            }

    return {"allowed": True, "reason": "Safe clarification", "category": "clarify"}


def build_chat_prompt(context: AuditContext, history: List[ChatMessage], user_text: str) -> str:
    """Constructs the system prompt, strictly grounding the LLM in the audit findings."""

    # Summarize flags for context
    flags_desc = []
    for f in context.flags:
        flags_desc.append(f"- {f.get('severity')} {f.get('category')}: {f.get('explanation')}. Evidence: {context.evidence_index.get(f'flag_{context.flags.index(f)}', '')[:100]}...")

    flags_block = "\n".join(flags_desc)

    system_prompt = f"""
You are the SentinelMD Safety Assistant.
Your precise role is to explain the Safety Audit findings to a clinician.

## STRICT RULES:
1. NO DIAGNOSIS: Do not diagnose conditions or interpret medical images.
2. NO TREATMENT: Do not recommend medications, dosages, or treatment plans.
3. GROUNDED ONLY: You can ONLY discuss the Flags and Missing Information listed below. If the user asks about something not in the audit, say you don't know.
4. KEYWORD: Use the term "Review" or "Consider" instead of "must" or "should".

## AUDIT CONTEXT:
Input Summary: {context.inputs_summary}

Safety Flags Found:
{flags_block}

Missing Info Questions:
{", ".join(context.missing_info)}

User Question: {user_text}

Answer concisely (3-4 sentences max). Use bullet points if listing evidence.
"""
    return system_prompt.strip()

def postprocess_answer(text: str) -> str:
    """Removes artifacts from LLM response."""
    clean = text.replace("Assistant:", "").strip()
    return clean
