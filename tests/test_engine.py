from src.core.engine import AnalysisEngine
from data.synthetic.generator import generate_patient_record, generate_safe_record

def test_penicillin_allergy_risk():
    record = generate_patient_record()
    engine = AnalysisEngine()
    report = engine.analyze(record)

    assert len(report.observations) >= 1
    # Check for Penicillin-specific findings
    penicillin_finding = next((obs for obs in report.observations if "Amoxicillin" in obs.evidence), None)
    assert penicillin_finding is not None
    assert penicillin_finding.severity == "HIGH"
    assert "Penicillin" in penicillin_finding.explanation

def test_safe_record():
    record = generate_safe_record()
    engine = AnalysisEngine()
    report = engine.analyze(record)

    assert len(report.observations) == 0
