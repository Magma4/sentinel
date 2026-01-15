"""
Lightweight text preprocessing to reduce prompt size before LLM calls.
Trimming is for context compression only - original text is preserved for evidence extraction.
"""
from typing import List


def trim_note(note: str) -> str:
    """
    Keeps only lines containing key clinical keywords plus surrounding context.
    Returns trimmed note for model context.
    """
    if not note:
        return note

    keywords = [
        "allerg", "nkda", "plan", "assessment", "discharge",
        "stable", "improving", "worsen", "aki", "ckd", "renal",
        "potassium", "k "
    ]

    lines = note.split('\n')
    kept_indices = set()

    # Find matching lines
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(kw in line_lower for kw in keywords):
            # Keep this line plus 3 lines before and after
            for j in range(max(0, i - 3), min(len(lines), i + 4)):
                kept_indices.add(j)

    if not kept_indices:
        # If no matches, keep first 10 lines as fallback
        return '\n'.join(lines[:10])

    # Reconstruct note with kept lines
    kept_lines = [lines[i] for i in sorted(kept_indices)]
    return '\n'.join(kept_lines)


def trim_labs(labs: str) -> str:
    """
    Keeps only lines containing key lab names.
    Returns trimmed labs for model context.
    """
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
    """
    Removes empty lines from medications list.
    Returns cleaned meds for model context.
    """
    if not meds:
        return meds

    lines = [line.strip() for line in meds.split('\n') if line.strip()]
    return '\n'.join(lines)
