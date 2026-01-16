from typing import Dict, Any, Optional
import re

try:
    import pypdf
except ImportError:
    pypdf = None

def extract_pdf_text(file_obj) -> str:
    """
    Parses text from a PDF file object using pypdf.
    Returns empty string on failure or if no text found.
    """
    if not pypdf:
        return "[Error: pypdf library not installed. Cannot process PDF.]"

    try:
        if hasattr(file_obj, "seek"): file_obj.seek(0)
        reader = pypdf.PdfReader(file_obj)
        text_parts = []
        for page in reader.pages:
            extract = page.extract_text()
            if extract:
                text_parts.append(extract)

        full_text = "\n".join(text_parts)
        if not full_text.strip():
            return "[Warning: No text extracted from PDF. It might be an image-only scan or encrypted.]"

        return full_text
    except Exception as e:
        return f"[Error parsing PDF: {str(e)}]"

def analyze_pdf_quality(text: str) -> Dict[str, Any]:
    """
    Analyzes extracted text for quality indicators.
    Returns a dict with metrics and a 'suspicious' flag.
    """
    # 1. Basic Stats
    clean_text = text.strip()
    char_count = len(clean_text)

    # 2. Heuristics for "Garbage" (e.g. encoding errors producing high non-ascii)
    non_ascii_count = sum(1 for c in clean_text if ord(c) > 127)
    non_ascii_ratio = non_ascii_count / char_count if char_count > 0 else 0

    # 3. Heuristics for "Empty/Sparse"
    # Clinical notes should rely heavily on standard alphanumeric + punctuation
    # Suspicious if > 30% non-ascii (arbitrary threshold for "garbage")
    is_garbage = non_ascii_ratio > 0.3

    # 4. Determine Warnings
    warnings = []

    if char_count < 100:
        warnings.append("Very short text extracted (<100 chars). May be incomplete.")
    elif char_count < 500:
        warnings.append("Short text extracted (<500 chars). Verify completeness.")

    if is_garbage:
        warnings.append("High ratio of non-standard characters detected. Encoding issue likely.")

    if "[Warning:" in text or "[Error:" in text:
        warnings.append("Extraction failed or returned errors.")

    return {
        "char_count": char_count,
        "non_ascii_ratio": round(non_ascii_ratio, 3),
        "suspicious_garbage": is_garbage,
        "is_short": char_count < 500,
        "warnings": warnings,
        "quality_pass": not is_garbage and char_count >= 100 and "[Error:" not in text
    }
