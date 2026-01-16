import pandas as pd
import io
import sys
from typing import Dict, Optional, Union, Any, List

# New Imports
from src.core.pdf_utils import extract_pdf_text

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

def parse_csv_labs(csv_content: Union[str, bytes]) -> str:
    """Parses standard lab CSVs (test/value columns) into text."""
    try:
        if isinstance(csv_content, bytes):
            # Try decoding as utf-8
            csv_content = csv_content.decode('utf-8', errors='replace')

        # Read CSV
        df = pd.read_csv(io.StringIO(csv_content))

        # Normalize columns
        df.columns = [c.lower().strip() for c in df.columns]

        col_map = {
            'test': ['test', 'name', 'test name', 'analyte', 'lab'],
            'value': ['value', 'result', 'reading'],
            'unit': ['unit', 'units'],
            'date': ['date', 'time', 'timestamp']
        }

        found_cols = {}
        for safe_key, candidates in col_map.items():
            for c in df.columns:
                if c in candidates:
                    found_cols[safe_key] = c
                    break

        # Minimum requirement: Test name and Value
        if 'test' not in found_cols or 'value' not in found_cols:
            return csv_content # Fallback to raw

        # Construct lines
        lines = []
        for _, row in df.iterrows():
            name = row[found_cols['test']]
            val = row[found_cols['value']]

            unit = f" {row[found_cols['unit']]}" if 'unit' in found_cols and pd.notna(row[found_cols['unit']]) else ""
            date = f" ({row[found_cols['date']]})" if 'date' in found_cols and pd.notna(row[found_cols['date']]) else ""

            lines.append(f"{name}: {val}{unit}{date}")

        return "\n".join(lines)

    except Exception as e:
        # On any error, return origin content to avoid crashing
        return str(csv_content)

def parse_image_text(file_obj) -> str:
    """Extracts text from images via Tesseract OCR."""
    if not Image:
        return "[Error: PIL/Pillow not installed.]"
    if not pytesseract:
        return "[Error: pytesseract not installed.]"

    try:
        if hasattr(file_obj, "seek"): file_obj.seek(0)
        img = Image.open(file_obj)

        try:
             text = pytesseract.image_to_string(img)
             return text if text.strip() else "[Warning: OCR found no text.]"
        except pytesseract.TesseractNotFoundError:
             return "[Error: Tesseract OCR binary not found on system. Please install Tesseract.]"
        except Exception as e:
             return f"[OCR Error: {str(e)}]"

    except Exception as e:
        return f"[Image Error: {str(e)}]"

def standardize_input(
    mode: str,
    note_input: Union[str, Any, List[Any]],
    labs_input: Union[str, Any, List[Any]],
    meds_input: Union[str, Any, List[Any]]
) -> Dict[str, str]:
    """Normalizes various input formats (Strings, Lists, Files) into a standard dictionary text block."""

    result = {
        "case_id": "USER_INPUT",
        "note_text": "",
        "labs_text": "",
        "meds_text": ""
    }

    # Helper to read different file types
    def read_single_item(inp) -> str:
        content = ""
        if inp is None:
            return ""
        if isinstance(inp, str):
            content = inp

        elif isinstance(inp, (list, tuple)):
            content = process_input(inp)

        # Streamlit UploadedFile
        elif hasattr(inp, "name"):
            filename = getattr(inp, "name", "").lower()

            # PDF
            if filename.endswith(".pdf"):
                content = extract_pdf_text(inp)

            # Image
            elif filename.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                content = parse_image_text(inp)

            # CSV (Special handling moved here)
            elif filename.endswith(".csv"):
                 if hasattr(inp, "seek"): inp.seek(0)
                 c_data = inp.read()
                 content = parse_csv_labs(c_data)

            # Default Text handling for BytesIO
            elif hasattr(inp, "read"):
                if hasattr(inp, "seek"): inp.seek(0)
                c_data = inp.read()
                if isinstance(c_data, bytes):
                    content = c_data.decode('utf-8', errors='replace')
                else:
                    content = c_data
            else:
                content = str(inp)
        else:
            content = str(inp)

        return content.strip()

    # Wrapper to handle Single Item vs List
    def process_input(inp_data) -> str:
        if inp_data is None:
            return ""


        # Robust check for any iterable (list, tuple, custom sequence)
        is_list_like = isinstance(inp_data, (list, tuple))
        if not is_list_like and hasattr(inp_data, "__iter__") and not isinstance(inp_data, (str, bytes, dict)):
            is_list_like = True

        if is_list_like:
            parts = []
            for item in inp_data:
                content = read_single_item(item)
                # Add separator for context if it's a file with a name
                name = getattr(item, "name", "")
                if name and len(list(inp_data)) > 1: # Cast to list to safely get len if needed
                     parts.append(f"--- Source: {name} ---\n{content}")
                else:
                     parts.append(content)
            return "\n\n".join(parts)
        else:
            return read_single_item(inp_data)

    # 1. Processing Logic
    result["note_text"] = process_input(note_input)
    result["labs_text"] = process_input(labs_input)
    result["meds_text"] = process_input(meds_input)

    return result
