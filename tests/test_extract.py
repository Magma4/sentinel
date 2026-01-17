from src.core.extract import FactExtractor

def test_mock_extraction_penicillin():
    extractor = FactExtractor(backend_url=None)

    note = "Patient has severe allergies to Penicillin."
    labs = "None"
    meds = "None"

    result = extractor.extract_facts(note, labs, meds)

    assert "Penicillin" in result["allergies"]
    assert result["patient_age"] == "UNKNOWN"

def test_mock_extraction_meds():
    extractor = FactExtractor(backend_url=None)

    note = "Started on Amoxicillin and Metformin."
    labs = ""
    meds = ""

    result = extractor.extract_facts(note, labs, meds)

    assert "Amoxicillin" in result["current_meds"]
    assert "Metformin" in result["current_meds"]
