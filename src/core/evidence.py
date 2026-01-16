from typing import List, Optional
from .schema import Evidence

def assert_quote_grounded(quote: str, source_text: str) -> bool:
    """Verifies that the quote exists exactly in the source_text."""
    return quote in source_text

def find_verbatim_quote(source_text: str, keywords: List[str]) -> Optional[str]:
    """
    Finds a relevant substring in source_text containing all keywords.
    Prioritizes single-line matches.
    """
    lower_source = source_text.lower()
    lower_keywords = [k.lower() for k in keywords]

    # 1. Try Line-based matching
    lines = source_text.split('\n')
    for line in lines:
        lower_line = line.lower()
        if all(k in lower_line for k in lower_keywords):
            return line.strip()

    # 2. Fallback: Whole text match if short
    if len(source_text) < 200:
        if all(k in lower_source for k in lower_keywords):
            return source_text.strip()

    return None

def build_evidence(source: str, source_text: str, quote: str, date: Optional[str] = None) -> Evidence:
    """
    Constructs an Evidence object after verifying grounding.
    Raises ValueError if quote is not found in source_text.
    """
    if not assert_quote_grounded(quote, source_text):
        # Relaxed check: Case-insensitive fallback
        idx = source_text.lower().find(quote.lower())
        if idx != -1:
            quote = source_text[idx : idx + len(quote)]
        else:
            # Failed to ground
             pass

    if quote not in source_text:
         raise ValueError(f"Quote '{quote}' not found in source '{source}'. Evidence must be verbatim.")

    # Auto-generate highlight
    import re
    numeric_pattern = r'\b\d+(\.\d+)?(/ \d+)?\b'
    def repl(match):
        return f"<b>{match.group(0)}</b>"

    highlighted = re.sub(numeric_pattern, repl, quote)

    return Evidence(
        quote=quote,
        highlighted_text=highlighted,
        source=source,
        source_date=date
    )
