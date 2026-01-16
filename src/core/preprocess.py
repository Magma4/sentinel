"""
Lightweight text preprocessing for prompt context optimization.
"""
from typing import List


def trim_note(note: str) -> str:
    """Retains only clinical keywords + context (3 lines before/after) to save tokens."""
    if not note:
        return note

    keywords = [
        "allerg", "nkda", "plan", "assessment", "discharge",
        "stable", "improving", "worsen", "aki", "ckd", "renal",
        "potassium", "k "
    ]

    lines = note.split('\n')
    kept_indices = set()

    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(kw in line_lower for kw in keywords):
            # Keep context window
            for j in range(max(0, i - 3), min(len(lines), i + 4)):
                kept_indices.add(j)

    if not kept_indices:
        # Fallback: Head of note
        return '\n'.join(lines[:10])

    kept_lines = [lines[i] for i in sorted(kept_indices)]
    return '\n'.join(kept_lines)


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
