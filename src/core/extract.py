from .schema import PatientRecord

def extract_facts_from_text(text: str) -> dict:
    """
    Placeholder: Extracts structured facts (meds, allergies) from raw clinical text.
    In future phases, this will use an LLM with 'prompts/extract_facts.md'.
    """
    # TODO: Implement LLM extraction logic
    return {}

def parse_record(raw_data: dict) -> PatientRecord:
    """
    Parses raw data into a validated PatientRecord.
    """
    return PatientRecord(**raw_data)
