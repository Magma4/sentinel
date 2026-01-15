from typing import List, Optional
from .schema import Evidence

def assert_quote_grounded(quote: str, source_text: str) -> bool:
    """
    Verifies that the quote exists exactly in the source_text (case-insensitive check,
    but we usually want strict exact match if possible. For robustness, we check case-insensitive presence
    and then return True, but strictly speaking 'verbatim' implies exact.
    Let's enforce case-insensitive for flexibility in keyword search, but the quote itself
    should ideally be extracted from the text, so strict check is better if we are generating it.)

    Update: The prompt asked for 'verbatim quote', implying exact substring.
    """
    return quote in source_text

def find_verbatim_quote(source_text: str, keywords: List[str]) -> Optional[str]:
    """
    Finds a relevant substring in source_text containing all keywords.
    Strategy:
    1. Identify lines or sentences containing ALL keywords.
    2. Return the first match.
    3. If no single line/sentence matches, try to find the shortest window?
    For simplicity: Check per-line (splitting by newline) first, then basic sliding window or just None.
    """
    lower_source = source_text.lower()
    lower_keywords = [k.lower() for k in keywords]

    # 1. Try Line-based matching
    lines = source_text.split('\n')
    for line in lines:
        lower_line = line.lower()
        if all(k in lower_line for k in lower_keywords):
            return line.strip()

    # 2. Fallback: If keywords denote a key-value pair usually close together (e.g. "Creatinine" "1.7")
    # Finding a small window might be complex for a helper. check if the whole text is short?
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
        # Retry with case-insensitive search to be kind, then replace with exact substring
        idx = source_text.lower().find(quote.lower())
        if idx != -1:
            quote = source_text[idx : idx + len(quote)]
        else:
            # If still not found, we can't accept it as verbatim
            # However, for the mock, we might want to fail gracefully or log warn.
            # The user requirement said: "ensure every evidence quote is a substring"
            # So let's fallback to strict mode or error?
            # In a real app, we might fallback to "Context: ..." but for this task:
            # "assert_quote_grounded" implies assertion logic.
            # Let's clean the quote (strip) just in case
            pass

    if quote not in source_text:
         # Final check
         raise ValueError(f"Quote '{quote}' not found in source '{source}'. Evidence must be verbatim.")

    # Auto-generate highlight if not provided
    highlighted = quote

    # Simple heuristic: Bold numbers and critical units
    import re
    # Find numbers like 5.9, 140, 120/80
    numeric_pattern = r'\b\d+(\.\d+)?(/ \d+)?\b'
    # Find keywords? This is harder without knowing the "finding".
    # But numbers are usually the "value" in "Creatinine 1.7".

    def repl(match):
        return f"<b>{match.group(0)}</b>"

    highlighted = re.sub(numeric_pattern, repl, quote)

    return Evidence(
        quote=quote,
        highlighted_text=highlighted,
        source=source,
        source_date=date
    )
