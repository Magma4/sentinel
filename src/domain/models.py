from enum import Enum, auto
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import hashlib
import time

# --- Enums ---

class SafetySeverity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"

class SafetyCategory(str, Enum):
    MEDICATION_INTERACTION = "MEDICATION_INTERACTION"
    CONTRAINDICATION = "CONTRAINDICATION"
    MISSING_MONITORING = "MISSING_MONITORING"
    DOSAGE_ERROR = "DOSAGE_ERROR"
    ALLERGY = "ALLERGY"
    CLINICAL_MISMATCH = "CLINICAL_MISMATCH"
    OTHER = "OTHER"

class ChatState(Enum):
    IDLE = auto()
    AUDIT_READY = auto()
    VALIDATE_QUERY = auto()
    REFUSE = auto()
    BUILD_CONTEXT = auto()
    ANSWERING = auto()
    RENDER_ANSWER = auto()
    ERROR = auto()

# --- Pydantic Models (Audit) ---

class EvidenceQuote(BaseModel):
    source: str  # Note, Labs, or Meds
    quote: str
    context: Optional[str] = None

class SafetyFlag(BaseModel):
    category: SafetyCategory
    severity: SafetySeverity
    explanation: str
    recommendation: str
    evidence: List[EvidenceQuote]
    reasoning: Optional[str] = None  # Chain-of-thought trace

class AuditReport(BaseModel):
    summary: str
    flags: List[SafetyFlag]
    missing_info_questions: List[str]
    confidence_score: float
    metadata: Optional[Dict[str, Any]] = None

# --- Pydantic Models (Patient Data) ---

class ClinicalNote(BaseModel):
    date: str
    author: str
    content: str
    type: str = "Progress Note"

class LabResult(BaseModel):
    name: str
    value: str # String allowing flexible formats
    unit: str
    date: str

class PatientRecord(BaseModel):
    patient_id: str
    notes: List[ClinicalNote] = []
    labs: List[LabResult] = []
    medications: List[str] = []
    allergies: List[str] = []


# --- Dataclasses (Chat + State) ---

@dataclass
class ChatMessage:
    role: str # user | assistant
    content: str
    ts: float = field(default_factory=time.time)

@dataclass
class AuditContext:
    audit_json: Dict
    flags: List[Dict]
    missing_info: List[str]
    inputs_summary: str
    evidence_index: Dict[str, str] # Map: flag_id -> quote

@dataclass
class ChatSession:
    state: ChatState = ChatState.IDLE
    history: List[ChatMessage] = field(default_factory=list)
    last_error: Optional[str] = None
    audit_fingerprint: str = ""
    context: Optional[AuditContext] = None

# --- Helpers ---

def fingerprint_audit(audit_json: Dict) -> str:
    """Hash audit data to detect state changes."""
    return hashlib.md5(str(audit_json).encode()).hexdigest()
