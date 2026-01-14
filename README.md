# SentinelMD

SentinelMD is an offline clinical safety copilot designed for the MedGemma Impact Challenge. It reviews patient records to flag potential safety risks, inconsistencies, and missing workflow steps.

## Core Constraints (Non-Negotiable)

- **Advisory Only**: NO diagnosis, NO treatment, NO medication changes. The system serves purely as a safety check.
- **Evidence-Based**: All outputs must be grounded in quoted evidence from the inputs.
- **Structured Output**: Output must be structured JSON, never free text.
- **Synthetic Data Only**: Do not use real patient data.
- **Offline-First**: No cloud APIs or external dependencies for core logic.

## Tech Stack

- **Language**: Python
- **Frontend**: Streamlit
- **Validation**: Pydantic
- **Data**: Synthetic datasets

## Getting Started

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run src/app/main.py
   ```
