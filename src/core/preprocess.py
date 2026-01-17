"""
Lightweight text preprocessing for prompt context optimization.
"""
from typing import List


def trim_note(note: str) -> str:
    """
    Intelligently prunes clinical notes to retain safety-critical sections.
    Uses regex to identify Assessment/Plan, Meds, and History while dropping administrative filler.
    """
    import re
    if not note: return ""

    # 1. Define Critical Headers (Regex)
    # Matches: "Assessment", "Plan", "A/P", "HPI", "Meds", "Allergies"
    headers = r"(?i)^(assessment|plan|recommendation|medication|allerg|hpi|history of present illness|impression|diagnosis)"

    lines = note.split('\n')
    path = []

    # Simple Heuristic: If note is short (< 50 lines), keep all.
    if len(lines) < 50:
        return note

    # 2. Block Extraction
    # Strategy: Keep first 10 lines (Metadata), then scan for blocks.
    path.extend(lines[:10])

    in_critical_block = False

    for line in lines[10:]:
        clean_line = line.strip()
        if re.match(headers, clean_line):
            in_critical_block = True
            path.append(f"\n--- {clean_line} ---") # Explicit separator for LLM
        elif not clean_line:
            pass # Skip empty lines unless in block? No, keep spacing for read.

        # Heuristic stop words for blocks (e.g. "Signed by", "Dictated by")
        if re.search(r"(?i)(signed by|dictated by|electronically signed)", clean_line):
            in_critical_block = False

        if in_critical_block:
             path.append(line)
        else:
            # Context preservation (keep if contains critical keywords)
            if any(kw in clean_line.lower() for kw in ["mg", "daily", "tabs", "allergic", "reaction", "ckd", "aki", "k+"]):
                path.append(line)

    result = "\n".join(path)

    # 3. Fallback check
    if len(result) < 100: # If we pruned too much
        return note[:2000] # Return head

    return result


def trim_labs(labs: str) -> str:
    """Filters lab report to high-priority safety analytes only."""
    if not labs:
        return labs

    key_labs = [
        "egfr", "creatin", "potassium", "k ", "ast", "alt",
        "inr", "wbc", "bun", "lactate"
    ]

    lines = labs.split('\n')
    kept_lines = []

    for line in lines:
        line_lower = line.lower()
        if any(lab in line_lower for lab in key_labs):
            kept_lines.append(line)

    return '\n'.join(kept_lines) if kept_lines else labs


def trim_meds(meds: str) -> str:
    """Normalizes medication list by removing empty lines."""
    if not meds:
        return meds

    lines = [line.strip() for line in meds.split('\n') if line.strip()]
    return '\n'.join(lines)
