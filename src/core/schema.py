from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# --- Enums ---

class SafetySeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class SafetyCategory(str, Enum):
    RISK = "RISK"
    INCONSISTENCY = "INCONSISTENCY"
    MISSING_STEP = "MISSING_STEP"

# --- Clinical Data Models ---

class ClinicalNote(BaseModel):
    """Represents a single clinical note entry."""
    date: str
    author: str
    content: str

class PatientRecord(BaseModel):
    """Complete patient record aggregated from various sources."""
    patient_id: str
    age: int
    gender: str
    allergies: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    vitals: Dict[str, Any] = Field(default_factory=dict)
    notes: List[ClinicalNote] = Field(default_factory=list)

# --- Audit/Safety Models ---

class SafetyObservation(BaseModel):
    """A specific safety finding based on evidence."""
    category: SafetyCategory
    severity: SafetySeverity
    evidence: str = Field(..., description="Verbatim quote from the patient record.")
    explanation: str
    recommendation: Optional[str] = Field(None, description="Advisory only.")

class SafetyReport(BaseModel):
    """Container for all observations for a specific patient."""
    patient_id: str
    observations: List[SafetyObservation] = Field(default_factory=list)
    summary: str
