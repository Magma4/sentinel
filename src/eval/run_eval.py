import json
import os
from src.core.schema import PatientRecord
from src.core.audit import SafetyAuditor
from src.eval.metrics import calculate_recall

def run_eval():
    """
    Runs the safety audit against all synthetic cases and prints metrics.
    """
    data_dir = "data/synthetic"
    auditor = SafetyAuditor()

    print("Starting Evaluation...")

    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r") as f:
            data = json.load(f)

        record = PatientRecord(**data)
        report = auditor.audit(record)

        print(f"Reviewed {filename}: Found {len(report.observations)} issues.")
        # TODO: Compare against ground truth labels (if they existed in the JSON)

if __name__ == "__main__":
    run_eval()
