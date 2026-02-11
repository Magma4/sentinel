# SentinelMD 🛡️

**The Offline Clinical Safety Copilot**

> *"Turning Chaos into Safety."*

SentinelMD is an offline-first "Edge AI" copilot for clinicians. It acts as a second pair of eyes, automatically cross-referencing clinical notes, medication lists, and lab results to detect life-threatening errors—like drug interactions and allergy conflicts—without a single byte of data leaving the device.

---

## 🚀 Key Features

### 1. **Real-Time Safety Audit** 🧠
Powered by **MedGemma-2-9b (Quantized)** running locally.
*   **Instant Risk Analysis:** Flags drug-drug interactions, drug-lab conflicts, and allergy mismatches.
*   **Evidence Grounding:** Every alert is backed by "Evidence Quotes" from the patient record to prevent hallucinations.
*   **Sequential Progress:** Live feedback on validation, DDI checks, and LLM analysis.

### 2. **Voice-to-Chart Dictation** 🎙️
*   **Offline Speech Recognition:** Uses **Whisper Large-v3** (via Apple MLX or Faster-Whisper) to transcribe clinical dictation instantly.
*   **Auto-Structuring:** Automatically extracts and structures unstructured voice notes into **Clinical Note**, **Medications**, and **Labs** fields.

### 3. **Population Health Dashboard** 📊
*   **Clinic-Wide Intelligence:** Aggregates risk data across all patient records.
*   **Risk Stratification:** Visualizes high-risk patients and common safety concerns (e.g., "Hyperkalemia Clusters").
*   **Offline Analytics:** All dashboards are generated locally from file-system data.

### 4. **Multimodal Ingestion** 📄
*   **Universal Upload:** Drag & Drop PDFs, Images (Scanned Labs), or Text files.
*   **On-Device OCR:** Digitizes paper records instantly using Tesseract.

---

## 🏗️ Architecture

SentinelMD follows a **Local-First, Service-Oriented Architecture**:

```mermaid
graph TD
    User([Clinician]) <-->|Interacts| UI[Streamlit Frontend]

    subgraph "Application Layer"
        UI -->|File Upload| InputLoader[Input Loader]
        UI -->|Audio| TranscriptionService[Transcription Service]
        TranscriptionService -->|Raw Text| FactExtractor[Fact Extractor]
        UI -->|Safety Check| AuditService[Audit Service]
        UI -->|CRUD & Stats| PatientService[Patient Service]
    end

    subgraph "Local Intelligence - Edge"
        InputLoader -->|PDF/Image| OCR[Tesseract OCR + PyPDF]
        TranscriptionService -->|Speech-to-Text| Whisper["Whisper - MLX / Faster-Whisper"]
        FactExtractor -->|Structuring| Ollama[Ollama Server]
        AuditService -->|Clinical Reasoning| Ollama
        AuditService -->|Deterministic Check| DDIChecker[DDI Checker]
        Ollama -.->|Weights| MedGemma[MedGemma 4B]
    end

    subgraph "Data Layer - File System"
        PatientService <-->|Read/Write JSON| Storage[(Local Patient DB)]
        Storage -->|Aggregation| PopHealth[Population Dashboard]
    end
```

| Component | Role | Module |
| :--- | :--- | :--- |
| **Input Loader** | Ingests PDFs, Images (OCR), CSVs, Text files | `src/core/input_loader.py` |
| **Transcription Service** | Offline speech-to-text (Whisper Large-v3) | `src/services/transcription_service.py` |
| **Fact Extractor** | LLM-powered structuring of raw text into Meds/Labs/Notes | `src/core/extract.py` |
| **Audit Service** | Orchestrates safety review (DDI + LLM analysis) | `src/services/audit_service.py` |
| **DDI Checker** | Deterministic drug-drug interaction database | `src/core/ddi_checker.py` |
| **Patient Service** | Patient CRUD, encounter history, population stats | `src/services/patient_service.py` |
| **Ollama / MedGemma** | Local LLM inference (quantized, on-device) | `src/core/llm_client.py` |

---

## 🛠️ Getting Started

### Prerequisites
1.  **Python 3.10+**
2.  **Ollama**: Install from [ollama.com](https://ollama.com).
3.  **Tesseract OCR**:
    *   Mac: `brew install tesseract`
    *   Linux: `sudo apt-get install tesseract-ocr`

### Installation
1.  **Clone the repo**:
    ```bash
    git clone https://github.com/Magma4/sentinel.git
    cd sentinel
    ```

2.  **Install Dependencies**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Pull the Model**:
    ```bash
    ollama pull amsaravi/medgemma-4b-it:q6
    ```

4.  **Run the App**:
    ```bash
    streamlit run src/app/ui_streamlit.py
    ```

---

## ⚠️ Disclaimer
**SentinelMD is a clinical decision support tool, NOT a diagnostic device.**
It is designed to identify potential documentation errors and safety risks for human review. It does not replace professional medical judgment.
