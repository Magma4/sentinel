"""
SentinelMD - Clinical Safety Audit Configuration & UI
=====================================================

This module serves as the main entry point for the Streamlit-based frontend.
It provides a user-friendly clinical dashboard that:
1.  Ingests multimodal patient data (Notes, Labs, Meds via PDF/Images/Text).
2.  Orchestrates the 'Safety Audit' via `AuditService` and `ReviewEngineAdapter`.
3.  Displays structured safety flags with evidence grounding.
4.  Hosts the 'Interactive Safety Assistant' (RAG-based chat).

Architecture:
-------------
- **Frontend**: Streamlit (Reactive UI).
- **Backbone**: `InputLoader` (Parsing) -> `AuditService` (Logic) -> `OllamaAdapter` (Inference).
- **State**: Manages session state for uploaded files, audit results, and chat history.
"""
import streamlit as st
import json
import os
import sys
import time
import hashlib

# Project Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from typing import Dict, Any, Optional
import importlib
import logging

# Module Imports
import src.domain.models
# importlib.reload(src.domain.models) # Hot-reload schema removed for production stability

from src.adapters.ollama_adapter import ReviewEngineAdapter
from src.core.input_loader import standardize_input
from src.services.audit_service import AuditService
from src.services.image_quality_service import ImageQualityService
from src.services.chat_service import ChatService
from src.domain.models import ChatSession, ChatMessage, PatientRecord, SafetyFlag
from src.domain.models import ChatSession, ChatMessage, PatientRecord, SafetyFlag
from src.eval.run_eval import run_eval_pipeline
from src.services.transcription_service import TranscriptionService
import io

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sentinel.ui")

# App Config
st.set_page_config(
    page_title="SentinelMD",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Theme Adaptation: Native Variables */
    :root {
        /* Severity Colors (RGBA for contrast) */
        --sev-high-bg: rgba(255, 75, 75, 0.1);    /* Red tint */
        --sev-high-border: #ff4b4b;

        --sev-med-bg: rgba(255, 164, 33, 0.1);    /* Orange tint */
        --sev-med-border: #ffa421;

        --sev-low-bg: rgba(33, 195, 84, 0.1);     /* Green tint */
        --sev-low-border: #21c354;

        --evidence-bg: var(--secondary-background-color);
        --evidence-border: var(--secondary-background-color);
        --evidence-text: var(--text-color);

        /* Badges */
        --badge-note: rgba(13, 71, 161, 0.1);
        --badge-labs: rgba(74, 28, 28, 0.1);
        --badge-meds: rgba(27, 94, 32, 0.1);

        /* Chat */
        --user-bubble-bg: var(--primary-color);
        --user-bubble-text: #ffffff;

        /* Text */
        --text-std: var(--text-color);
        --text-muted: gray;
    }

    .main .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }

    /* Severity Block */
    .severity-block {
        padding: 10px;
        border-radius: 0 5px 5px 0;
        margin-bottom: 10px;
        color: var(--text-std);
    }

    .severity-high {
        border-left: 5px solid var(--sev-high-border);
        background-color: var(--sev-high-bg);
    }
    .severity-medium {
        border-left: 5px solid var(--sev-med-border);
        background-color: var(--sev-med-bg);
    }
    .severity-low {
        border-left: 5px solid var(--sev-low-border);
        background-color: var(--sev-low-bg);
    }

    .evidence-block {
        background-color: var(--evidence-bg);
        border-left: 3px solid var(--text-muted);
        padding: 8px;
        margin: 5px 0;
        font-family: monospace;
        font-size: 0.9em;
        color: var(--evidence-text);
        border-radius: 4px;
    }

    .user-bubble {
        background-color: var(--user-bubble-bg);
        color: var(--user-bubble-text);
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin-right: 10px;
        max-width: 75%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    .summary-card {
        padding: 15px;
        border-radius: 8px;
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 20px;
        color: var(--text-std);
    }

    .chat-history-box {
        height: 400px;
        overflow-y: auto;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 15px;
        background-color: var(--secondary-background-color);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🛡️ SentinelMD")
    st.caption("Clinical Safety Copilot")
    st.markdown("---")

    # Workflow Selection
    input_mode_map = {
        "Load Reference Case": "Demo Cases",
        "Clinical Dictation": "Paste Text",
        "Upload Medical Records": "Upload Files"
    }

    workflow = st.radio(
        "Workflow",
        list(input_mode_map.keys()),
        help="Select clinical workflow."
    )
    input_mode = input_mode_map[workflow]

    st.markdown("---")

    # Case Selector (if applicable)
    selected_case_file = None
    if input_mode == "Demo Cases":
        DATA_DIR = "data/synthetic"
        cases = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])
        display_cases = [c.replace(".json", "").replace("_"," ").title() for c in cases]
        sel_idx = st.selectbox("Select Patient Record", display_cases, index=0)
        selected_case_file = cases[display_cases.index(sel_idx)]

    # Advanced / Technical Section (Collapsed by default)
    with st.expander("🛠️ Advanced Configuration", expanded=False):
        # 1. Review Engine
        model_map = {
            "Standard Safety Engine": "amsaravi/medgemma-4b-it:q6",
            "Test Mode (Mock)": "mock-model"
        }

        display_name = st.selectbox(
            "Inference Engine",
            options=list(model_map.keys()),
            index=0
        )
        selected_model = model_map[display_name]
        st.session_state.backend_type = "ollama"

        # Init Services (Silent)
        # Force re-init to pick up new methods
        if "audit_service" not in st.session_state or st.session_state.current_model != selected_model or True: # Force True for hotfix
             try:
                import importlib
                import src.services.audit_service as audit_svc_module
                import src.adapters.ollama_adapter as ollama_adapt_module

                # Force reload modules to ensure new methods are picked up
                importlib.reload(audit_svc_module)
                importlib.reload(ollama_adapt_module)

                # Re-import classes from reloaded modules
                from src.services.audit_service import AuditService
                from src.adapters.ollama_adapter import ReviewEngineAdapter

                adapter = ReviewEngineAdapter(model=selected_model)
                if not adapter.check_connection():
                     st.error("⚠️ Engine Offline. Run `ollama serve`.")
                else:
                     pass

                st.session_state.audit_service = AuditService(adapter)
                st.session_state.chat_service = ChatService(adapter)
                st.session_state.current_model = selected_model
                st.session_state.inference_cache = {}
             except Exception as e:
                st.error(f"Init Failed: {e}")

        st.caption("Privacy: 100% Offline | OCR: Local")

        # Options
        st.session_state.one_call_mode = st.checkbox("Fast Mode", value=False)
        st.session_state.max_flags = 10

        # Metrics
        if "last_report" in st.session_state:
            report = st.session_state.last_report
            ext_time = report.metadata.get("extract_runtime", 0)
            aud_time = report.metadata.get("audit_runtime", 0)
            st.text(f"Latency: {(ext_time + aud_time):.2f}s")

    # Simple Status Indicator
    st.success("✅ System Ready (Secure)")

# Main Layout
st.warning("⚠️ **ADVISORY ONLY**: Identifying safety risks. Not a substitute for clinical judgment.")

# Global Inputs
record: Optional[PatientRecord] = None
standardized_inputs: Dict[str, str] = {
    "case_id": "UNKNOWN",
    "note_text": "",
    "labs_text": "",
    "meds_text": ""
}
file_source_type = "TEXT"
quality_report = None

# Input Handling
if input_mode == "Demo Cases":
    file_source_type = "DEMO"
    # Patient Banner
    pt_id_str = selected_case_file.replace(".json", "").replace("_"," ").replace("Pt Record", "").strip()
    if pt_id_str.startswith("Mrn"): pt_id_str = pt_id_str.replace("Mrn", "MRN:")

    st.markdown(f"""
    <div style="background-color: transparent; border-bottom: 2px solid #f0f2f6; padding-bottom: 10px; margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2.2rem;">Patient Record</h1>
        <p style="color: #666; font-size: 1.1rem; margin: 0;">Case ID: <b>{pt_id_str}</b></p>
    </div>
    """, unsafe_allow_html=True)

    current_case_path = os.path.join(DATA_DIR, selected_case_file)
    with open(current_case_path, "r") as f:
        case_data = json.load(f)
        record = PatientRecord(**case_data)

    # Load from Record
    standardized_inputs = {
        "case_id": selected_case_file.replace(".json", ""),
        "note_text": "\n".join([n.content for n in record.notes]),
        "labs_text": "\n".join([f"{l.name}: {l.value} {l.unit}" for l in record.labs]),
        "meds_text": ", ".join(record.medications),
        "images": []
    }

elif input_mode == "Paste Text":
    st.header("Clinical Documentation")
    st.caption("Dictate or paste clinical notes below.")

    col_p1, col_p2, col_p3 = st.columns(3)

    # Voice Dictation (Optional - populates Clinical Note)
    st.caption("💡 **Voice Input**: Speak your full clinical note including medications and labs. The AI will parse everything automatically.")
    audio_val = st.audio_input("🎙️ Dictate")
    if audio_val:
        # User Confirmation Step
        st.info("Audio captured. Click below to process.")

        if st.button("✨ Transcribe Audio", type="primary", key="transcribe_btn"):
            st.session_state.last_audio = audio_val # Update state to track processed audio
            with st.spinner("Transcribing... (Using Local/Edge Model)"):
                try:
                    # Initialize Service (Singleton pattern via session_state)
                    # Renamed key to force re-init after code changes
                    if "transcription_service_mlx" not in st.session_state:
                         st.session_state.transcription_service_mlx = TranscriptionService(model_size="large-v3")

                    # Streamlit audio_input gives a BytesIO-like object.
                    # faster-whisper needs a file path or binary stream.
                    # We'll save it to a temp file to be safe and compatible.
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_val.read())
                        tmp_path = tmp_file.name

                    # Transcribe with Medical Context Prompt
                    # This primes the model to output clinical terminology.
                    medical_prompt = "Clinical note. Patient history, symptoms, medications, interactions, diagnosis, cardiology, oncology, daily dosage."
                    text = st.session_state.transcription_service_mlx.transcribe(tmp_path, initial_prompt=medical_prompt)

                    # Cleanup
                    os.remove(tmp_path)

                    st.session_state.note_in_val = text
                    st.toast("✅ Transcription Complete!", icon="🎙️")
                    st.rerun()  # Force clean redraw to prevent UI ghosting

                except Exception as e:
                    st.error(f"Transcription Error: {e}")

    # Sync Text Area with Session State
    if "note_in_val" not in st.session_state:
        st.session_state.note_in_val = ""

    # Check if voice was used (transcription exists)
    has_voice_input = bool(st.session_state.note_in_val.strip())

    if has_voice_input:
        # Unified view for voice dictation
        st.markdown("### 📝 Your Dictation")
        st.info("💡 Your voice recording has been transcribed below. All medications, labs, and clinical context mentioned will be analyzed together.")
        note_in = st.text_area("Full Clinical Context", height=250, key="note_in_val",
                               help="Edit your dictation if needed. The AI will parse medications and labs automatically.")
        labs_in = ""
        meds_in = ""
    else:
        # Structured input view (no voice yet)
        with col_p1:
            note_in = st.text_area("Clinical Note", height=300,
                                 placeholder="Example:\nPt presented with weakness...\nPMH: CKD, HTN...",
                                 key="note_in_val")
        with col_p2:
            labs_in = st.text_area("Labs", height=300, placeholder="Example:\nPotassium: 6.0 mmol/L\nCreatinine: 2.1 mg/dL")
        with col_p3:
            meds_in = st.text_area("Medications", height=300, placeholder="Example:\nLisinopril 10mg daily\nSpironolactone 25mg daily")

    # Bypass Adapter (Raw Strings)
    standardized_inputs = {
        "note_text": note_in,
        "labs_text": labs_in,
        "meds_text": meds_in,
        "quality_report": []
    }
    standardized_inputs["case_id"] = "USER_PASTE"
    standardized_inputs["images"] = []

elif input_mode == "Upload Files":
    st.header("Medical Record Import")
    st.caption("Import PDF, CSV, or Image files. Patient context is automatically unified.")

    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        note_files = st.file_uploader("Clinical Notes", type=["txt", "pdf", "png", "jpg", "jpeg", "csv", "json"], accept_multiple_files=True)
        if note_files:
            file_source_type = "MULTIPLE FILES" if len(note_files) > 1 else f"FILE ({note_files[0].name.split('.')[-1].upper()})"
            for f in note_files:
                 if f.name.lower().endswith('.pdf'): file_source_type = "PDF" # Flag as PDF if any PDF present
            st.caption(f"Selected: {len(note_files)} file(s)")
            for f in note_files:
                 st.caption(f"- {f.name}")

    with col_u2:
        labs_files = st.file_uploader("Labs", type=["txt", "pdf", "png", "jpg", "jpeg", "csv", "json"], accept_multiple_files=True)
        if labs_files:
            st.caption(f"Selected: {len(labs_files)} file(s)")
            for f in labs_files: st.caption(f"- {f.name}")

    with col_u3:
        meds_files = st.file_uploader("Medications", type=["txt", "pdf", "png", "jpg", "jpeg", "csv", "json"], accept_multiple_files=True)
        if meds_files:
            st.caption(f"Selected: {len(meds_files)} file(s)")
            for f in meds_files: st.caption(f"- {f.name}")

    standardized_inputs = standardize_input("UPLOAD", note_files, labs_files, meds_files)
    standardized_inputs["case_id"] = "USER_UPLOAD"
    standardized_inputs["quality_report"] = []

    # Deterministic Image Check
    all_files = (note_files or []) + (labs_files or []) + (meds_files or [])
    for f in all_files:
        if f.name.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                f.seek(0)
                file_bytes = f.read()
                img = ImageQualityService.load_image(file_bytes)
                q_res = ImageQualityService.compute_quality(img)
                q_res["filename"] = f.name
                standardized_inputs["quality_report"].append(q_res)
            except Exception as e:
                st.error(f"Failed to analyze image {f.name}: {e}")

    # (PDF Logic kept as is for now)

    # --- PROGRESSIVE UI: Immediate Feedback ---
    if note_files or labs_files or meds_files:
        n_len = len(standardized_inputs["note_text"].split())
        l_len = len(standardized_inputs["labs_text"].splitlines())
        m_len = len(standardized_inputs["meds_text"].split(',')) if standardized_inputs["meds_text"] else 0

        # Simple extraction heuristics for display
        st.markdown("### ⚡ Data Extraction Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Clinical Note", f"{n_len} words", delta="Ready" if n_len > 0 else "Empty")
        k2.metric("Lab Items", f"{l_len} lines", delta="Ready" if l_len > 0 else "Empty")
        k3.metric("Medications", f"{m_len} items", delta="Ready" if m_len > 0 else "Empty")

        q_pass = True
        if "quality_report" in standardized_inputs and standardized_inputs["quality_report"]:
             q_issues = sum(len(qr['quality_issues']) for qr in standardized_inputs["quality_report"])
             q_pass = q_issues == 0

        k4.metric("Scan Quality", "Optimal" if q_pass else "Review Needed", delta_color="normal" if q_pass else "inverse")
        st.info("👆 Review the patient summary above. If correct, proceed to **Safety Review**.")
        st.divider()
    # ------------------------------------------

# Tabs
tab_safety, tab_inputs, tab_eval = st.tabs(["🛡️ Safety Analysis", "📄 Patient Record", "📋 History & Export"])

# Tab 1: Safety Review
with tab_safety:
    col_act, col_info = st.columns([1, 4])
    with col_act:
        run_btn = st.button("▶️ Run Safety Review", type="primary", use_container_width=True, key="btn_run_safety")

    if run_btn:
        # Validation
        if not standardized_inputs["note_text"] and not standardized_inputs["meds_text"] and not standardized_inputs["labs_text"]:
            st.error("❌ No input data detected. Please provide Note, Labs, or Medications.")
        else:
            # Cache Key
            backend_str = st.session_state.backend_type
            model_str = st.session_state.audit_service.engine.model
            input_payload = (standardized_inputs["note_text"] + standardized_inputs["labs_text"] + standardized_inputs["meds_text"]).encode()
            input_hash = hashlib.md5(input_payload).hexdigest()[:8]
            cache_key = f"{standardized_inputs['case_id']}_{input_hash}_{backend_str}_{model_str}"

            if "inference_cache" not in st.session_state:
                st.session_state.inference_cache = {}

            # Check Cache
            if cache_key in st.session_state.inference_cache:
                cached_data = st.session_state.inference_cache[cache_key]
                st.session_state.last_report = cached_data["report"]
                st.info("⚡ Analysis loaded from cache")
            else:
                with st.status("🚀 Running Safety Review (Local Engine)...", expanded=True) as status:
                    t0 = time.time()
                    # status.write("Extracting clinical entities...") # Optional progress steps

                    note_text = standardized_inputs["note_text"]
                    labs_text = standardized_inputs["labs_text"]
                    meds_text = standardized_inputs["meds_text"]

                    # Audit Execution
                    report = st.session_state.audit_service.run_safety_review(
                        note_text, labs_text, meds_text
                    )

                    status.update(label="✅ Analysis Complete", state="complete", expanded=False)

                    if report:
                        st.session_state.last_report = report
                        st.session_state.inference_cache[cache_key] = {"report": report}

                        # Track in session history
                        if "review_history" not in st.session_state:
                            st.session_state.review_history = []
                        from datetime import datetime
                        st.session_state.review_history.append({
                            "timestamp": datetime.now().strftime("%I:%M %p"),
                            "case_id": standardized_inputs.get("case_id", "Unknown"),
                            "input_preview": (note_text[:80] + "...") if len(note_text) > 80 else note_text,
                            "flag_count": len(report.flags),
                            "max_severity": max([f.severity.value if hasattr(f.severity, 'value') else str(f.severity) for f in report.flags], default="NONE"),
                            "report": report
                        })

                        st.rerun()  # Force clean redraw to prevent ghosting
                    else:
                        st.error("Safety Review Warning: Engine execution failed.")

    if "last_report" in st.session_state:
        report = st.session_state.last_report

        # Summary Card
        flag_count = len(report.flags)
        max_severity = "NONE"
        if flag_count > 0:
            order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            # Handle Pydantic Enums vs Strings
            def get_sev(x): return x.severity.value if hasattr(x.severity, 'value') else str(x.severity)

            sorted_flags = sorted(report.flags, key=lambda x: order.get(get_sev(x), 0), reverse=True)
            max_severity = get_sev(sorted_flags[0])

        # Dashboard Metrics
        st.markdown(f"### Audit Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Safety Flags", flag_count, delta="Issues Found" if flag_count > 0 else "Clean", delta_color="inverse")
        m2.metric("Highest Severity", max_severity, delta="Risk Level" if max_severity!="NONE" else "Safe", delta_color="inverse")
        m3.metric("Review Confidence", f"{report.confidence_score*100:.0f}%", "AI Estimate")

        st.info(f"**Analysis**: {report.summary}")



        # Flags Display
        if not report.flags:
            st.success("✅ No safety issues detected from available inputs. Verify completeness.")
        else:
            for flag in report.flags:
                # Styles
                sev_val = flag.severity.value if hasattr(flag.severity, 'value') else str(flag.severity)
                css_class = f"severity-{sev_val.lower()}"

                cat_val = flag.category.value if hasattr(flag.category, 'value') else str(flag.category)
                display_cat = cat_val.replace("_", " ").title()
                if display_cat == "Other":
                    display_cat = "General Safety Constraint"
                elif display_cat == "Medication Interaction":
                    display_cat = "Medication-Allergy Conflict"

                with st.container():
                    # Layout: Explanation (Let) | Buttons (Right)
                    f_col1, f_col2 = st.columns([0.85, 0.15])

                    with f_col1:
                        st.markdown(f"""
                        <div class="{css_class} severity-block">
                            <h4>[{sev_val}] {display_cat}</h4>
                            <p><b>{flag.explanation}</b></p>
                        </div>
                        """, unsafe_allow_html=True)

                    with f_col2:
                         # Feedback Buttons
                         safe_key = hashlib.md5(flag.explanation.encode()).hexdigest()[:8]

                         def save_feedback(rating, expl, cat):
                             import csv
                             from datetime import datetime

                             fb_dir = os.path.join(project_root, "data", "feedback")
                             os.makedirs(fb_dir, exist_ok=True)
                             fb_file = os.path.join(fb_dir, "user_feedback.csv")

                             file_exists = os.path.isfile(fb_file)

                             with open(fb_file, "a", newline="") as f:
                                 writer = csv.writer(f)
                                 if not file_exists:
                                     writer.writerow(["timestamp", "case_id", "category", "explanation", "rating"])

                                 writer.writerow([
                                     datetime.now().isoformat(),
                                     standardized_inputs.get("case_id", "UNKNOWN"),
                                     cat,
                                     expl,
                                     rating
                                 ])
                             st.toast(f"Feedback Saved: {rating}!", icon="💾")

                         if st.button("👍", key=f"up_{safe_key}", help="This flag is helpful/accurate"):
                             save_feedback("HELPFUL", flag.explanation, display_cat)

                         if st.button("👎", key=f"down_{safe_key}", help="False Positive / Not Useful"):
                             save_feedback("FALSE_POSITIVE", flag.explanation, display_cat)

                    with st.expander("Show Evidence", expanded=True):
                        st.caption("Verbatim quotes from record:")
                        for ev in flag.evidence:
                             quote = ev.quote
                             src = ev.source

                             badge_color = "var(--evidence-bg)"
                             if src == "NOTE": badge_color = "var(--badge-note)"
                             elif src == "LABS": badge_color = "var(--badge-labs)"
                             elif src == "MEDS": badge_color = "var(--badge-meds)"


                             st.markdown(f"""
                             <div class="evidence-block">
                                <span style="background-color: {badge_color}; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.8em; margin-right: 5px;">{src}</span>
                                "{quote}"
                             </div>
                             """, unsafe_allow_html=True)





        # ---------------------------------------------------------
        # 🗣️ Patient Translator (After-Activity Summary)
        # ---------------------------------------------------------
        st.divider()
        st.markdown("### 🗣️ Patient Translator")

        pt_lang = st.radio("Language", ["🇺🇸 English", "🇪🇸 Spanish"], horizontal=True, label_visibility="collapsed")
        pt_lang_code = "English" if "English" in pt_lang else "Spanish"

        if st.button(f"Generate Patient Instructions ({pt_lang_code})", key="btn_pt_gen"):
            with st.spinner("Writing simple instructions..."):
                note_text = standardized_inputs["note_text"]

                # Extract Safety Flags Context
                safety_context = []
                if "last_report" in st.session_state and st.session_state.last_report:
                    # Simplify flags for context window efficiency
                    safety_context = [f"{f.category}: {f.explanation}" for f in st.session_state.last_report.flags]

                pt_data = st.session_state.audit_service.get_patient_instructions(note_text, pt_lang_code, safety_context)
                st.session_state.last_patient_instructions = pt_data

        if "last_patient_instructions" in st.session_state:
            pt = st.session_state.last_patient_instructions
            if "error" in pt:
                st.error(pt["error"])
            else:
                st.markdown(f"#### 📝 Take-Home Summary ({pt_lang_code})")
                st.info(f"**Doctor Note:** {pt.get('summary', '')}")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**📌 Key Takeaways**")
                    for k in pt.get("key_takeaways", []):
                        st.markdown(f"- {k}")

                with c2:
                    st.markdown("**💊 Medications**")
                    for m in pt.get("medication_instructions", []):
                        st.markdown(f"- {m}")

                # Medical Decoder
                if pt.get("terminology_map"):
                    with st.expander("📖 Medical Decoder (Terms Explained)", expanded=False):
                        for term in pt["terminology_map"]:
                            st.markdown(f"**{term.get('term')}** → _{term.get('simple')}_")


        st.divider()
        # Quality Review
        if "quality_report" in standardized_inputs and standardized_inputs["quality_report"]:
            all_issues = []
            for qr in standardized_inputs["quality_report"]:
                all_issues.extend(qr.get("quality_issues", []))

            if not all_issues:
                st.success("✅ **Visual Input Review**: Images appear readable (Quality Check Only)")
            else:
                st.warning(f"⚠️ **Visual Input Review**: {len(all_issues)} potential readability issues detected")

            with st.expander("Show Quality Details (Deterministic)", expanded=False):
                st.caption("Quality check only. No diagnosis. OCR used for content.")

                for qr in standardized_inputs["quality_report"]:
                    st.markdown(f"**{qr['filename']}**")
                    if qr["quality_issues"]:
                        for issue in qr["quality_issues"]: st.markdown(f"- 🔴 {issue}")
                    if qr["workflow_risks"]:
                        for risk in qr["workflow_risks"]: st.markdown(f"- 🟠 {risk}")
                    if not qr["quality_issues"] and not qr["workflow_risks"]:
                        st.markdown("- ✅ Readable")
                    for obs in qr.get("visual_observations", []):
                        st.markdown(f"- *{obs}*")
                    st.divider()

         # Missing Info
        questions = getattr(report, 'missing_info_questions', [])
        if questions:
            st.markdown("#### ❓ Missing Information / Clarifications")
            for q in questions:
                 st.info(q)
        # Assistant (Chat)
        # Assistant moved to floating popover

# Tab 2: Inputs
with tab_inputs:
    col1, col2 = st.columns(2)

    # Highlights
    highlights = []
    if "last_report" in st.session_state:
        for flag in st.session_state.last_report.flags:
            for ev in flag.evidence:
                quote = getattr(ev, 'quote', None) or (ev.get('quote') if isinstance(ev, dict) else str(ev))
                if quote: highlights.append(quote)

    with col1:
        st.subheader("Clinical Note")

        st.markdown(f"**Source**: `{file_source_type}`")
        if quality_report:
             color = "green" if quality_report['quality_pass'] else "red"
             st.markdown(f"**Quality Check**: <span style='color:{color}'>{quality_report['char_count']} chars, {int(quality_report['non_ascii_ratio']*100)}% non-standard</span>", unsafe_allow_html=True)
             if not quality_report['quality_pass']:
                 st.write(quality_report['warnings'])

        content = standardized_inputs["note_text"]

        # Highlight evidence
        display_content = content
        for h in highlights:
            if h in display_content:
                display_content = display_content.replace(h, f"<mark style='background-color: #fff3cd; color: #856404; font-weight: bold;'>{h}</mark>")

        st.markdown(f"""
        <div style="height: 500px; overflow-y: auto; background-color: white; color: #31333F; padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px; font-family: 'Source Sans Pro', sans-serif; white-space: pre-wrap; line-height: 1.6; font-size: 16px;">{display_content if display_content else "No clinical note provided."}</div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Structured Data")

        if record:
            # Demo View
            def highlight_matches(row):
                row_str = str(row.values)
                style = [''] * len(row)
                for h in highlights:
                    for i, cell in enumerate(row):
                         if str(cell) and (str(cell) in h or h in str(cell)):
                             style[i] = 'background-color: #fff3cd; color: #856404; font-weight: bold;'
                return style

            st.markdown("**Medications**")
            df_meds = pd.DataFrame(record.medications, columns=["Medication"])
            st.dataframe(df_meds.style.apply(highlight_matches, axis=1), hide_index=True, use_container_width=True)

            st.markdown("**Laboratories**")
            if record.labs:
                df_labs = pd.DataFrame([l.model_dump() for l in record.labs])
                st.dataframe(df_labs.style.apply(highlight_matches, axis=1), hide_index=True, use_container_width=True)
            else:
                st.caption("No labs recorded.")

            st.markdown("**Allergies**")
            st.write(", ".join(record.allergies) if record.allergies else "None/Unknown")

        else:
            # Text View (Styled)
            # Only show what we have
            has_meds = bool(standardized_inputs["meds_text"].strip())
            has_labs = bool(standardized_inputs["labs_text"].strip())

            if has_meds:
                st.markdown("**Medications**")
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #6c757d; margin-bottom: 15px; font-family: monospace;">
                    {standardized_inputs["meds_text"]}
                </div>
                """, unsafe_allow_html=True)

            if has_labs:
                st.markdown("**Laboratories**")
                st.markdown(f"""
                 <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #6c757d; margin-bottom: 15px; font-family: monospace;">
                    {standardized_inputs["labs_text"]}
                </div>
                """, unsafe_allow_html=True)

            if not has_meds and not has_labs:
                st.caption("No structured data (Meds/Labs) extracted.")

# Tab 3: History & Export
with tab_eval:
    st.subheader("📋 Session History")

    if "review_history" not in st.session_state or not st.session_state.review_history:
        st.info("No safety reviews yet this session. Run a review from the **Safety Analysis** tab to see history here.")
    else:
        from datetime import datetime

        for i, review in enumerate(reversed(st.session_state.review_history)):
            severity_color = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡", "NONE": "🟢"}.get(review["max_severity"], "⚪")
            report = review["report"]

            # Create short recognizable name from input
            input_words = review['input_preview'].split()[:5]  # First 5 words
            short_name = " ".join(input_words)
            if len(input_words) == 5:
                short_name += "..."

            # Generate export text for this specific review
            report_text = f"""SENTINEL MD - SAFETY REVIEW REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Case: {review['case_id']}
{'='*50}

SUMMARY
Confidence: {report.confidence_score*100:.0f}% | Flags: {len(report.flags)}
{report.summary}

{'='*50}
SAFETY FLAGS
"""
            for j, flag in enumerate(report.flags, 1):
                sev = flag.severity.value if hasattr(flag.severity, 'value') else str(flag.severity)
                cat = flag.category.value if hasattr(flag.category, 'value') else str(flag.category)
                report_text += f"""
[{j}] {sev} - {cat}
    {flag.explanation}
    Recommendation: {flag.recommendation if flag.recommendation else 'Review guidelines.'}
"""
            report_text += f"""
{'='*50}
DISCLAIMER: Advisory only. Consult healthcare professionals.
"""

            # Header row: Expander title + Download button side by side
            col_expand, col_dl = st.columns([5, 1])

            with col_dl:
                st.download_button(
                    label="⬇️",
                    data=report_text,
                    file_name=f"safety_report_{review['case_id']}_{datetime.now().strftime('%H%M')}.txt",
                    mime="text/plain",
                    key=f"dl_{i}",
                    help="Download this report as .txt"
                )

            with col_expand:
                with st.expander(f"{severity_color} **{review['timestamp']}** — \"{short_name}\" ({review['flag_count']} flag{'s' if review['flag_count'] != 1 else ''})", expanded=(i == 0)):
                    st.caption(f"**Case ID**: {review['case_id']} | **Confidence**: {report.confidence_score*100:.0f}%")

                    st.markdown(f"**Analysis**: {report.summary}")

                    if review["flag_count"] > 0:
                        st.divider()
                        for flag in report.flags:
                            sev = flag.severity.value if hasattr(flag.severity, 'value') else str(flag.severity)
                            cat = flag.category.value if hasattr(flag.category, 'value') else str(flag.category)
                            sev_style = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}.get(sev, "⚪")

                            st.markdown(f"{sev_style} **[{sev}] {cat.replace('_', ' ').title()}**")
                            st.markdown(f"> {flag.explanation}")
                            if flag.recommendation:
                                st.caption(f"💡 {flag.recommendation}")
                            st.markdown("---")
                    else:
                        st.success("✅ No safety issues detected")

# --- Floating Chat Implementation ---
def render_floating_chat(standardized_inputs):
    """
    Renders the Safety Assistant in a premium floating UI.
    Matches the "Purple Gradient" aesthetic requested.
    """

    # --- CSS STYLES ---
    st.markdown("""
    <style>
    /* 1. Floating Action Button Container */
    div[data-testid="stPopover"] {
        position: fixed !important;
        bottom: 30px !important;
        right: 30px !important;
        width: 60px !important; /* Force circle size */
        height: 60px !important;
        z-index: 9999 !important;
        background-color: transparent !important;
    }

    /* Target the button inside - Force it to be a circle */
    div[data-testid="stPopover"] > button {
        background: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%) !important;
        color: white !important;
        border-radius: 50% !important;
        width: 60px !important;
        height: 60px !important;
        box-shadow: 0 4px 15px rgba(74, 0, 224, 0.4) !important;
        border: none !important;
        font-size: 28px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: transform 0.2s !important;
    }

    div[data-testid="stPopover"] > button:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 6px 20px rgba(74, 0, 224, 0.6) !important;
    }

    /* Remove any default text/arrow from the button if possible (Streamlit adds '...') */
    div[data-testid="stPopover"] > button > div {
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* 2. Chat Bubbles */
    .chat-bubble-user {
        background: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px; /* Tail bottom-right */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 8px;
        font-size: 14px;
        line-height: 1.5;
        max-width: 85%;
        float: right;
        clear: both;
    }

    .chat-bubble-bot {
        background: #f1f2f6;
        color: #2c3e50;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px; /* Tail bottom-left */
        margin-bottom: 8px;
        font-size: 14px;
        line-height: 1.5;
        max-width: 85%;
        float: left;
        clear: both;
        border: 1px solid #e1e4e8;
    }

    /* 3. Chat Header (Simulated) */
    .chat-header {
        background: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%);
        padding: 15px;
        border-radius: 10px 10px 0 0;
        color: white;
        margin: -1rem -1rem 1rem -1rem; /* Negative margins to fill popover */
        display: flex;
        align-items: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .chat-header-avatar {
        font-size: 24px;
        margin-right: 12px;
        background: rgba(255,255,255,0.2);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .chat-header-info h4 {
        margin: 0;
        color: white;
        font-size: 16px;
        font-weight: 600;
    }
    .chat-header-info p {
        margin: 0;
        color: rgba(255,255,255,0.8);
        font-size: 12px;
    }

    </style>
    """, unsafe_allow_html=True)

    # The Popover
    with st.popover("💬", use_container_width=False):
            # Header
            st.markdown("""
            <div class="chat-header">
                <div class="chat-header-avatar">🛡️</div>
                <div class="chat-header-info">
                    <h4>Sentinel Assistant</h4>
                    <p>Safety & Compliance Bot</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # --- CHAT LOGIC ---
            if "chat_session" not in st.session_state:
                st.session_state.chat_session = ChatSession()

            report = st.session_state.get("last_report")

            if not report:
                st.markdown("""
                <div style="text-align: center; padding: 20px; color: #666;">
                    <br>
                    Run a <b>Safety Analysis</b> first to activate the bot.
                </div>
                """, unsafe_allow_html=True)
            else:
                 # Init Context if needed
                if not st.session_state.chat_session.history and hasattr(st.session_state, 'chat_service'):
                    audit_dict = report.model_dump()
                    raw_note = standardized_inputs.get('note_text', '')
                    input_summary_txt = f"Clinical Note Content:\n{raw_note[:2000]}"

                    st.session_state.chat_session = st.session_state.chat_service.reset_session(
                        st.session_state.chat_session,
                        audit_dict,
                        input_summary_txt
                    )

                # Render Chat History
                chat_cont = st.container(height=350)
                with chat_cont:
                    # History
                    for msg in st.session_state.chat_session.history:
                         if msg.role == "user":
                             st.markdown(f'<div class="chat-bubble-user">{msg.content}</div>', unsafe_allow_html=True)
                         else:
                             st.markdown(f'<div class="chat-bubble-bot">{msg.content}</div>', unsafe_allow_html=True)

                    # Spacer to ensure scrolling hits bottom
                    st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)

                # Input Area
                if query := st.chat_input("Type a message...", key="float_chat_premium"):
                    st.session_state.chat_session.history.append(ChatMessage(role="user", content=query))
                    st.rerun()

                # Generation
                if st.session_state.chat_session.history and st.session_state.chat_session.history[-1].role == "user":
                    with chat_cont:
                        with st.spinner("Thinking..."):
                             reply = st.session_state.chat_service.generate_reply(
                                 st.session_state.chat_session,
                                 st.session_state.chat_session.history[-1].content
                             )
                             st.session_state.chat_session.history.append(ChatMessage(role="assistant", content=reply))
                             st.rerun()

# Call the function
render_floating_chat(standardized_inputs)
