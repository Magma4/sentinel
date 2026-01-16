from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import uuid
from datetime import datetime

# --- Enums ---

class SafetySeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class SafetyCategory(str, Enum):
    MED_LAB_CONFLICT = "MED_LAB_CONFLICT"
    TEMPORAL_CONTRADICTION = "TEMPORAL_CONTRADICTION"
    MISSING_WORKFLOW_STEP = "MISSING_WORKFLOW_STEP"
    DOC_INCONSISTENCY = "DOC_INCONSISTENCY"

# --- Clinical Data Models ---

class ClinicalNote(BaseModel):
    """Represents a single clinical note entry."""
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    author: str = "Unknown"
    content: str

class LabResult(BaseModel):
    name: str
    value: Union[float, int, str]
    unit: str
    date: str

class GroundTruthItem(BaseModel):
    category: str
    severity: str
    key: str

class PatientRecord(BaseModel):
    """Complete patient record aggregated from various sources."""
    patient_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    allergies: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list, alias="meds")
    labs: List[LabResult] = Field(default_factory=list)
    vitals: Dict[str, Any] = Field(default_factory=dict)
    notes: List[ClinicalNote] = Field(default_factory=list)
    ground_truth: List[GroundTruthItem] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def parse_note_string(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Handle 'note' -> 'notes' conversion
            if 'note' in data and not 'notes' in data:
                content = data.pop('note')
                if isinstance(content, str):
                    data['notes'] = [{'content': content, 'date': '2024-01-01', 'author': 'System'}]
        return data

# --- Audit/Safety Models ---

class Evidence(BaseModel):
    quote: str = Field(..., max_length=160, description="Verbatim substring from source text.")
    highlighted_text: Optional[str] = Field(None, description="HTML-safe string with key values emphasized.")
    source: Optional[str] = "UNKNOWN"
    source_date: Optional[str] = None
    page_number: Optional[int] = None

    @field_validator('quote', mode='before')
    @classmethod
    def validate_quote_length(cls, v: str) -> str:
        if len(v) > 160:
            # Enforce conciseness
            return v[:157] + "..."
        return v

class SafetyFlag(BaseModel):
    """A specific safety finding based on evidence."""
    category: SafetyCategory
    severity: SafetySeverity
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[Evidence] = Field(..., min_length=1)
    explanation: str
    reasoning: Optional[str] = Field(None, description="Detailed situation analysis.")
    recommendation: Optional[str] = Field(None, description="Advisory only.", validation_alias="review_guidance")

    @field_validator('explanation', 'recommendation')
    @classmethod
    def validate_safety_language(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v

        unsafe_terms = {"start", "stop"}
        tokens = set(v.lower().replace(".", "").replace(",", "").split())

        violations = unsafe_terms.intersection(tokens)
        if violations:
            raise ValueError(f"Unsafe directive language detected: {violations}. "
                             "Safety flags must be advisory only (no treatment changes).")
        return v

class SafetyReport(BaseModel):
    """Container for all observations for a specific patient."""
    patient_id: str
    flags: List[SafetyFlag] = Field(default_factory=list)
    summary: str
    missing_info_questions: List[str] = Field(default_factory=list, description="Questions clarifying gaps or ambiguities.")
    metadata: Dict[str, Any] = Field(default_factory=dict)
