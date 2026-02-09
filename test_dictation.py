
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.core.extract import FactExtractor

def test_dictation():
    print("🎙️ Testing Clinical Dictation Parser...\n")

    transcript = """
    Patient is James Wilson, a 55-year-old male presenting with chest pain.
    He has a history of hypertension and hyperlipidemia.
    Current medications include Lisinopril 10mg daily and Atorvastatin 40mg at bedtime.
    We are starting Aspirin 81mg daily.
    Labs today showed a Troponin of 0.04 and a Creatinine of 1.2.
    Plan is to admit for observation.
    """

    print(f"--- Input Transcript ---\n{transcript}\n")

    extractor = FactExtractor(backend_type="ollama", backend_url="http://localhost:11434")

    print("🧠 Parsing with MedGemma...")
    result = extractor.parse_dictation(transcript)

    print("\n--- Structured Output ---")
    print(json.dumps(result, indent=2))

    # Validation
    if "Lisinopril" in result["medications"] and "0.04" in result["labs"]:
        print("\n✅ PASS: Extracting Meds and Labs correctly.")
    else:
        print("\n❌ FAIL: Missing key elements.")

if __name__ == "__main__":
    test_dictation()
