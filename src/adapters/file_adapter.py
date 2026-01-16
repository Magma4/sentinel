import pandas as pd
import io
from typing import Dict, List, Any, Union
import logging
from src.core.pdf_utils import analyze_pdf_quality

logger = logging.getLogger("sentinel.adapters.files")

def standardize_input(
    note_files: List[Any],
    labs_files: List[Any],
    meds_files: List[Any]
) -> Dict[str, Any]:
    """Normalizes uploaded files (txt, csv, pdf) into text buffers."""

    std_data = {
        "note_text": "",
        "labs_text": "",
        "meds_text": "",
        "quality_report": []
    }

    def read_text(file_obj) -> str:
        try:
            if file_obj.name.endswith(".csv"):
                df = pd.read_csv(file_obj)
                return df.to_markdown(index=False)
            elif file_obj.name.endswith(".pdf"):
                import pypdf
                reader = pypdf.PdfReader(file_obj)
                text = []
                for page in reader.pages:
                     text.append(page.extract_text() or "")
                return "\n".join(text)
            else:
                stringio = io.StringIO(file_obj.getvalue().decode("utf-8"))
                return stringio.read()
        except Exception as e:
            logger.error(f"Error reading file {file_obj.name}: {e}")
            return f"[Error interpreting {file_obj.name}]"

    # Process Notes
    if note_files:
        texts = [read_text(f) for f in note_files if not f.name.endswith(('.png', '.jpg', '.jpeg'))]
        std_data["note_text"] = "\n\n".join(texts)

    # Process Labs
    if labs_files:
        texts = [read_text(f) for f in labs_files]
        std_data["labs_text"] = "\n\n".join(texts)

    # Process Meds
    if meds_files:
        texts = [read_text(f) for f in meds_files]
        std_data["meds_text"] = "\n\n".join(texts)

    return std_data
