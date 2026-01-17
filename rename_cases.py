import os
import json
import re

data_dir = "data/synthetic"

def get_context(data):
    # Try to infer a short context string from conditions or notes
    conditions = data.get("conditions", [])
    if conditions:
        # Take first condition, sanitize
        c = conditions[0]
        return re.sub(r'[^a-zA-Z0-9]', '', c)[:15]

    # Check notes for keywords
    notes = data.get("notes", [])
    if notes:
        content = notes[0].get("content", "").lower()
        if "discharge" in content: return "Discharge"
        if "allergy" in content: return "Allergy"
        if "lab" in content: return "LabReview"

    return "General"

files = sorted([f for f in os.listdir(data_dir) if f.startswith("case_") and f.endswith(".json")])

for f in files:
    old_path = os.path.join(data_dir, f)
    try:
        with open(old_path, 'r') as json_file:
            data = json.load(json_file)

        # Extract ID number
        # case_001.json -> 001
        num = f.split('_')[1].split('.')[0]

        context = get_context(data)
        new_name = f"Pt_Record_MRN_{num}_{context}.json"

        new_path = os.path.join(data_dir, new_name)

        # Renaissance
        os.rename(old_path, new_path)
        print(f"Renamed: {f} -> {new_name}")

    except Exception as e:
        print(f"Skipped {f}: {e}")
