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
from src.services.patient_service import PatientService
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

# Patient Service Init
if "patient_service" not in st.session_state:
    st.session_state.patient_service = PatientService()

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
        "Patient Records": "Patient Records",
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
    if input_mode == "Patient Records":
        st.sidebar.info("🔍 Search or Create Patient")

        # Load Patients
        all_patients = st.session_state.patient_service.get_all_patients()
        patient_options = ["+ New Patient"] + [f"{p['name']} ({p['dob']})" for p in all_patients]

        # Sync Selectbox with State
        default_ix = None
        if st.session_state.get("current_patient"):
            curr = st.session_state.current_patient
            fmt = f"{curr['name']} ({curr['dob']})"
            if fmt in patient_options:
                default_ix = patient_options.index(fmt)

        sel_idx = st.sidebar.selectbox("Patient Record", patient_options, index=default_ix, placeholder="Search or Select Patient...")

        if sel_idx == "+ New Patient":
            with st.sidebar.form("new_pat_wf"):
                st.markdown("### 🆕 New Patient Profile")
                n_name = st.text_input("Full Name", placeholder="e.g. Jane Doe")
                n_dob = st.text_input("DOB (YYYY-MM-DD)", placeholder="e.g. 1980-01-01")
                n_mrn = st.text_input("MRN (Optional)", placeholder="e.g. MRN-12345")

                submitted = st.form_submit_button("Create Profile", type="primary")

                if submitted:
                    import re
                    # Validation
                    if not n_name:
                        st.error("Name is required.")
                    elif not re.match(r"\d{4}-\d{2}-\d{2}", n_dob):
                        st.error("Invalid DOB format. Use YYYY-MM-DD.")
                    else:
                        # Proceed with creation
                        new_p = st.session_state.patient_service.create_patient(n_name, n_dob, mrn=n_mrn if n_mrn else None)
                        if new_p is None:
                            st.error(f"⚠️ Patient '{n_name}' with DOB '{n_dob}' already exists!")
                        else:
                            st.session_state.current_patient = new_p
                            st.toast(f"Profile Created: {n_name}", icon="✨")
                            st.rerun()
            st.session_state.current_patient = None

        elif sel_idx:
            # Find patient object
            found_p = None
            for p in all_patients:
                if f"{p['name']} ({p['dob']})" == sel_idx:
                    found_p = p
                    break

            if found_p:
                # Detect patient change: Only load encounter data if we're switching patients
                prev_patient_id = st.session_state.get("_loaded_patient_id", None)
                is_new_patient = (prev_patient_id != found_p["id"])

                st.session_state.current_patient = found_p
                st.sidebar.success(f"Active: {found_p['name']}")
                st.sidebar.caption(f"MRN: {found_p.get('mrn','')}")

                # Only load encounter data if this is a NEW patient selection
                if is_new_patient:
                    # Mark this patient as loaded so we don't overwrite edits on rerun
                    st.session_state._loaded_patient_id = found_p["id"]

                    # Check for existing encounters to load
                    encounters = st.session_state.patient_service.get_encounters(found_p["id"])
                    if encounters:
                        # Load the latest
                        latest = encounters[0]
                        # Note: save_encounter stores under 'inputs' key
                        input_data = latest.get("inputs", latest.get("input_data", {}))

                        # Populate inputs
                        st.session_state.note_in_val = input_data.get("note", "")
                        st.session_state.meds_in_val = input_data.get("meds", "")
                        st.session_state.labs_in_val = input_data.get("labs", "")

                        st.toast(f"Loaded records for {found_p['name']}", icon="📂")
                    else:
                        # New patient with no records - clear fields
                        st.session_state.note_in_val = ""
                        st.session_state.meds_in_val = ""
                        st.session_state.labs_in_val = ""
                        st.toast(f"Active: {found_p['name']} (No records yet)", icon="👤")

                # Delete Patient Button (always visible when patient is selected)
                st.sidebar.divider()
                with st.sidebar.expander("⚠️ Danger Zone", expanded=False):
                    st.warning(f"Permanently delete **{found_p['name']}** and all their records?")
                    col_del1, col_del2 = st.columns(2)
                    with col_del1:
                        if st.button("🗑️ Delete", key="btn_delete_patient", type="primary"):
                            st.session_state._confirm_delete = True
                    with col_del2:
                        if st.session_state.get("_confirm_delete"):
                            if st.button("✅ Confirm", key="btn_confirm_delete"):
                                success = st.session_state.patient_service.delete_patient(found_p["id"])
                                if success:
                                    st.session_state.current_patient = None
                                    st.session_state._loaded_patient_id = None
                                    st.session_state._confirm_delete = False
                                    st.toast(f"Deleted {found_p['name']}", icon="🗑️")
                                    st.rerun()
                                else:
                                    st.error("Delete failed")

            pass
        else:
             st.session_state.current_patient = None

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
# Input Handling
if input_mode == "Patient Records":
    # 0. Define Tabs EARLY (Layout Change)
    tab_safety, tab_inputs, tab_eval = st.tabs(["🛡️ Safety Analysis", "📄 Patient Record", "📋 History & Export"])

    with tab_inputs:
        # 1. Header & Edit Toggle
        p_name = st.session_state.current_patient['name'] if st.session_state.current_patient else "Guest User"

        col_hdr, col_btn = st.columns([4, 1])
        with col_hdr:
             st.markdown(f"""
            <div style="background-color: transparent; border-bottom: 2px solid #f0f2f6; padding-bottom: 10px; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2.2rem;">Patient Record</h1>
                <p style="color: #666; font-size: 1.1rem; margin: 0;">Active: <b>{p_name}</b></p>
            </div>
            """, unsafe_allow_html=True)

        # Init Edit State
        if "rec_edit_mode" not in st.session_state: st.session_state.rec_edit_mode = False

        # Helper to init temp keys
        def init_temp_keys():
             st.session_state.note_tmp = st.session_state.get("note_in_val", "")
             st.session_state.meds_tmp = st.session_state.get("meds_in_val", "")
             st.session_state.labs_tmp = st.session_state.get("labs_in_val", "")

        with col_btn:
             if not st.session_state.rec_edit_mode:
                 if st.button("✏️ Edit Record", type="secondary", key="btn_enter_edit"):
                     init_temp_keys()
                     st.session_state.rec_edit_mode = True
                     st.rerun()
             else:
                 st.caption("✨ Editing Mode")

        # 2. Init Data State (Ensures defaults exist if not yet set)
        if "note_in_val" not in st.session_state: st.session_state.note_in_val = ""
        if "meds_in_val" not in st.session_state: st.session_state.meds_in_val = ""
        if "labs_in_val" not in st.session_state: st.session_state.labs_in_val = ""

        # Init Temp State if missing (safety fallback)
        if "note_tmp" not in st.session_state: st.session_state.note_tmp = ""
        if "meds_tmp" not in st.session_state: st.session_state.meds_tmp = ""
        if "labs_tmp" not in st.session_state: st.session_state.labs_tmp = ""

        # 3. Layout: 3 Columns
        col_e1, col_e2, col_e3 = st.columns(3)

        # Determine Content based on State
        if st.session_state.rec_edit_mode:
            # --- Clinical Note ---
            with col_e1:
                st.markdown("### 📝 Clinical Note")
                note_in = st.text_area("Edit Content", height=200, key="note_tmp", placeholder="Enter clinical narrative...")

            # --- Labs ---
            with col_e2:
                st.markdown("### 🧪 Labs")
                labs_in = st.text_area("Edit Content", height=200, key="labs_tmp", placeholder="Potassium: 4.5...")

            # --- Meds ---
            with col_e3:
                st.markdown("### 💊 Medications")
                meds_in = st.text_area("Edit Content", height=200, key="meds_tmp", placeholder="Lisinopril 10mg...")

            # Action Buttons (Footer)
            st.divider()
            c_save, c_disc = st.columns([1, 6])
            with c_save:
                 if st.button("Save Changes", type="primary", key="btn_save_edit_footer"):
                     try:
                         # 1. Extract content from Uploads
                         u_notes = st.session_state.get("rec_u_notes")
                         u_labs = st.session_state.get("rec_u_labs")
                         u_meds = st.session_state.get("rec_u_meds")

                         ext_data = standardize_input("UPLOAD", u_notes, u_labs, u_meds)

                         # 2. Merge Text
                         # Fallback to session state if note_in is empty but tmp exists?
                         src_n = note_in if note_in else st.session_state.get("note_tmp", "")
                         src_l = labs_in if labs_in else st.session_state.get("labs_tmp", "")
                         src_m = meds_in if meds_in else st.session_state.get("meds_tmp", "")

                         final_n = (src_n + "\n\n" + ext_data["note_text"]).strip()
                         final_l = (src_l + "\n\n" + ext_data["labs_text"]).strip()
                         final_m = (src_m + "\n\n" + ext_data["meds_text"]).strip()

                         # 3. Commit to Persistent State
                         st.session_state.note_in_val = final_n
                         st.session_state.labs_in_val = final_l
                         st.session_state.meds_in_val = final_m

                         # 4. PERSIST TO DISK (for survival across refreshes)
                         curr_patient = st.session_state.get("current_patient")
                         if curr_patient:
                             input_data = {
                                 "note": final_n,
                                 "labs": final_l,
                                 "meds": final_m
                             }
                             st.session_state.patient_service.save_encounter(
                                 patient_id=curr_patient["id"],
                                 input_data=input_data,
                                 report_data={}  # No report yet, just saving inputs
                             )

                         # 5. Exit Method
                         st.session_state.rec_edit_mode = False
                         st.toast("Record Saved to Disk", icon="💾")
                         st.rerun()

                     except Exception as e:
                         st.error(f"Save Failed: {e}")
            with c_disc:
                 if st.button("Discard", type="secondary", key="btn_discard_edit_footer"):
                     st.session_state.rec_edit_mode = False
                     st.rerun()

            # 4. Global File Attachments (Only in Edit Mode)
            with st.expander("📎 Attach Documents (PDF/Images)", expanded=True):
                st.info("Files uploaded here are processed and appended.")
                col_u1, col_u2, col_u3 = st.columns(3)
                with col_u1:
                    note_files = st.file_uploader("Note Files", type=["txt", "csv", "json", "pdf", "png", "jpg", "jpeg", "docx", "xlsx"], accept_multiple_files=True, key="rec_u_notes")
                with col_u2:
                    labs_files = st.file_uploader("Lab Files", type=["txt", "csv", "json", "pdf", "png", "jpg", "jpeg", "docx", "xlsx"], accept_multiple_files=True, key="rec_u_labs")
                with col_u3:
                    meds_files = st.file_uploader("Med Files", type=["txt", "csv", "json", "pdf", "png", "jpg", "jpeg", "docx", "xlsx"], accept_multiple_files=True, key="rec_u_meds")


        else:
            # Read-Only View (Hides the input fields)
            # We initialize input vars from state to ensure pipeline continuity
            note_in = st.session_state.note_in_val
            labs_in = st.session_state.labs_in_val
            meds_in = st.session_state.meds_in_val
            note_files, labs_files, meds_files = None, None, None

            # Show a nice summary or nothing?
            # User said "Show these only when toggled".
            # We should probably show the nice "Structured Data" view here?
            # For now, let's show a placeholder or the same "No content" view but simpler?
            # Actually, let's just show the Text/Labs/Meds nicely formatted like before but FULL WIDTH?
            # or just rely on the bottom section?
            # The User's screenshot shows duplicates. If I hide the top part, the bottom part remains.
            # So hiding the top part achieves "Show these only when toggled".

            # We'll render the formatted view here to be safe/useful
            # We'll render the formatted view here to be safe/useful
            if not note_in and not labs_in and not meds_in:
                 st.markdown("""
                 <div style="text-align: center; padding: 40px; background-color: #f8f9fa; border-radius: 10px; border: 1px dashed #ccc;">
                    <h3 style="color: #6c757d; margin-bottom: 10px;">📭 Patient Record is Empty</h3>
                    <p style="color: #6c757d;">No clinical data has been recorded yet.</p>
                    <p style="color: #495057; font-size: 0.9em;">Click the <b>✏️ Edit Record</b> button above to start documenting.</p>
                 </div>
                 """, unsafe_allow_html=True)
            else:
                 c1, c2, c3 = st.columns(3)

                 # Helper style
                 box_style = "background-color:#ffffff; padding:15px; border-radius:8px; border: 1px solid #e9ecef; box-shadow: 0 1px 3px rgba(0,0,0,0.05); max-height:400px; overflow-y:auto;"

                 with c1:
                     st.markdown("### 📝 Clinical Note")
                     content = note_in if note_in else '<em style="color:#aaa">No note recorded.</em>'
                     st.markdown(f"<div style='{box_style}'>{content}</div>", unsafe_allow_html=True)
                 with c2:
                     st.markdown("### 🧪 Labs")
                     content = labs_in if labs_in else '<em style="color:#aaa">No labs recorded.</em>'
                     st.markdown(f"<div style='{box_style} white-space: pre-wrap;'>{content}</div>", unsafe_allow_html=True)
                 with c3:
                     st.markdown("### 💊 Medications")
                     content = meds_in if meds_in else '<em style="color:#aaa">No medications recorded.</em>'
                     st.markdown(f"<div style='{box_style} white-space: pre-wrap;'>{content}</div>", unsafe_allow_html=True)


        # 5. Pipeline Handoff (Common Logic)
        upload_inputs = standardize_input("UPLOAD", note_files, labs_files, meds_files)

        # If files were just uploaded, we might want to automatically append them to the text?
        # But standardize_input returns the TEXT extracted from them.
        # Logic: If we are in Edit Mode, we combine current text + new file text.
        # But if we leave them in upload_inputs, they get "re-appended" every frame?
        # No, standardize_input re-reads the file every frame.
        # If we append to note_in (the variable), it doesn't update session_state.note_in_val automatically.
        # We rely on the final `standardized_inputs` dict construction to downstream users.

        final_note = note_in + ("\n\n" + upload_inputs["note_text"] if upload_inputs["note_text"] else "")
        final_labs = labs_in + ("\n\n" + upload_inputs["labs_text"] if upload_inputs["labs_text"] else "")
        final_meds = meds_in + ("\n\n" + upload_inputs["meds_text"] if upload_inputs["meds_text"] else "")

        # Optional: Deterministic Image List
        img_q_list = []
        all_files = (note_files or []) + (labs_files or []) + (meds_files or [])
        for f in all_files:
            if f.name.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    f.seek(0)
                    file_bytes = f.read()
                    img = ImageQualityService.load_image(file_bytes)
                    q_res = ImageQualityService.compute_quality(img)
                    q_res["filename"] = f.name
                    img_q_list.append(q_res)
                except: pass

        standardized_inputs = {
            "case_id": p_name,
            "note_text": final_note.strip(),
            "labs_text": final_labs.strip(),
            "meds_text": final_meds.strip(),
            "quality_report": img_q_list # Use the new img_q_list
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
    if "note_in_val" not in st.session_state: st.session_state.note_in_val = ""
    if "meds_in_val" not in st.session_state: st.session_state.meds_in_val = ""
    if "labs_in_val" not in st.session_state: st.session_state.labs_in_val = ""

    # Check if voice was used (transcription exists)
    has_voice_input = bool(st.session_state.note_in_val.strip())

    if has_voice_input:
        # Unified view for voice dictation
        st.markdown("### 📝 Your Dictation")
        st.info("💡 Your voice recording has been transcribed below. All medications, labs, and clinical context mentioned will be analyzed together.")

        # Init widget from state
        if "widget_paste_note" not in st.session_state:
             st.session_state.widget_paste_note = st.session_state.get("note_in_val", "")

        note_in = st.text_area("Full Clinical Context", height=250, key="widget_paste_note",
                               help="Edit your dictation if needed. The AI will parse medications and labs automatically.")
        # Sync back
        st.session_state.note_in_val = note_in

        labs_in = ""
        meds_in = ""
    else:
        # Structured input view (no voice yet)
        with col_p1:
             # Init widget
             if "widget_paste_note_struct" not in st.session_state: st.session_state.widget_paste_note_struct = st.session_state.get("note_in_val", "")

             note_in = st.text_area("Clinical Note", height=300,
                                  placeholder="Example:\nPt presented with weakness...\nPMH: CKD, HTN...",
                                  key="widget_paste_note_struct")
             st.session_state.note_in_val = note_in

        with col_p2:
             if "widget_paste_labs" not in st.session_state: st.session_state.widget_paste_labs = st.session_state.get("labs_in_val", "")
             labs_in = st.text_area("Labs", height=300, placeholder="Example:\nPotassium: 6.0 mmol/L\nCreatinine: 2.1 mg/dL", key="widget_paste_labs")
             st.session_state.labs_in_val = labs_in

        with col_p3:
             if "widget_paste_meds" not in st.session_state: st.session_state.widget_paste_meds = st.session_state.get("meds_in_val", "")
             meds_in = st.text_area("Medications", height=300, placeholder="Example:\nLisinopril 10mg daily\nSpironolactone 25mg daily", key="widget_paste_meds")
             st.session_state.meds_in_val = meds_in

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

        # --- AUTO-DETECT PATIENT FROM UPLOADED FILES ---
        st.divider()
        st.markdown("### 👤 Patient Identification")

        # Try to detect patient name from extracted content or filename
        detected_name = None
        detected_dob = None

        # Check uploaded file names for patient info patterns
        for f in all_files:
            fname = f.name.lower()
            # Simple heuristic: if filename contains "patient_" or a name pattern
            if "_" in fname:
                parts = fname.replace(".pdf", "").replace(".txt", "").replace(".csv", "").split("_")
                if len(parts) >= 2:
                    # Try to parse as "LastName_FirstName" or "Name_DOB"
                    potential_name = " ".join(parts[:2]).title()
                    if len(potential_name) > 3 and not potential_name.isdigit():
                        detected_name = potential_name
                        break

        # Check content for "Patient:" or "Name:" patterns using regex
        import re
        content = standardized_inputs["note_text"]
        name_match = re.search(r'(?:Patient|Name|Patient Name)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)', content)
        if name_match:
            detected_name = name_match.group(1).strip()

        dob_match = re.search(r'(?:DOB|Date of Birth|Birth Date)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})', content)
        if dob_match:
            detected_dob = dob_match.group(1).strip()

        # UI for patient creation
        if detected_name:
            st.success(f"📋 Detected patient: **{detected_name}**" + (f" (DOB: {detected_dob})" if detected_dob else ""))

        col_p1, col_p2, col_p3 = st.columns([2, 2, 1])
        with col_p1:
            input_name = st.text_input("Patient Name", value=detected_name or "", placeholder="e.g., John Smith", key="upload_patient_name")
        with col_p2:
            input_dob = st.text_input("Date of Birth", value=detected_dob or "", placeholder="e.g., 1980-01-01", key="upload_patient_dob")
        with col_p3:
            st.write("") # Spacer
            st.write("") # Spacer
            if st.button("➕ Create Patient", key="btn_create_from_upload", type="primary", disabled=not input_name):
                if input_name:
                    new_p = st.session_state.patient_service.create_patient(input_name, input_dob or "Unknown")
                    if new_p is None:
                        st.error(f"⚠️ Patient '{input_name}' with DOB '{input_dob or 'Unknown'}' already exists!")
                    else:
                        st.session_state.current_patient = new_p
                        st.session_state._loaded_patient_id = new_p["id"]

                        # Also populate the record with extracted content
                        st.session_state.note_in_val = standardized_inputs["note_text"]
                        st.session_state.labs_in_val = standardized_inputs["labs_text"]
                        st.session_state.meds_in_val = standardized_inputs["meds_text"]

                        st.toast(f"Created patient: {input_name}", icon="✅")
                        st.rerun()

        st.divider()
    # ------------------------------------------

# Tabs
if input_mode != "Patient Records":
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

                        # --- AUTO-DETECT PATIENT ---
                        # If the LLM found demographics, try to match or create patient
                        if report.patient_demographics and "name" in report.patient_demographics:
                            detected_name = report.patient_demographics["name"]
                            detected_dob = report.patient_demographics.get("dob", "Unknown")

                            # Only act if we aren't already locked onto a specific patient (or are in Guest/New mode)
                            # Actually, even if locked, we might want to warn? For now, assume Guest/New.
                            current_p = st.session_state.get("current_patient")

                            if not current_p:
                                # Search existing
                                all_p = st.session_state.patient_service.get_all_patients()
                                match = None
                                for p in all_p:
                                    if p["name"].lower() == detected_name.lower(): # Simple match
                                        match = p
                                        break

                                if match:
                                    st.session_state.current_patient = match
                                    st.toast(f"Matched existing patient: {detected_name}", icon="🔗")
                                else:
                                    # Create new (only if not duplicate)
                                    new_p = st.session_state.patient_service.create_patient(detected_name, detected_dob)
                                    if new_p is None:
                                        st.warning(f"Patient '{detected_name}' already exists - skipping auto-create")
                                    else:
                                        st.session_state.current_patient = new_p
                                        st.toast(f"Auto-created patient: {detected_name}", icon="✨")

                                # Rerun so the sidebar updates? Only if we want immediate visual feedback.
                                # But we need to save the encounter first!

                        # ---------------------------

                        # --- RECORD KEEPING ---
                        if "current_patient" in st.session_state and st.session_state.current_patient:
                            pat_id = st.session_state.current_patient["id"]
                            rpt_data = {
                                "summary": report.summary,
                                "flags": [{"category": str(f.category), "severity": str(f.severity), "explanation": f.explanation} for f in report.flags],
                                "confidence": report.confidence_score
                            }
                            st.session_state.patient_service.save_encounter(
                                patient_id=pat_id,
                                input_data={
                                    "note": standardized_inputs["note_text"],
                                    "meds": standardized_inputs["meds_text"],
                                    "labs": standardized_inputs["labs_text"]
                                },
                                report_data=rpt_data
                            )
                            st.toast(f"Saved to {st.session_state.current_patient['name']}", icon="💾")

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
if input_mode != "Patient Records":
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
    st.subheader("📋 Session History & Patient Records")

    # If a patient is selected, show their permanent history first
    if "current_patient" in st.session_state and st.session_state.current_patient:
        curr = st.session_state.current_patient
        st.markdown(f"#### 🗂️ Record: {curr['name']} (DOB: {curr['dob']})")

        # Load encounters
        encounters = st.session_state.patient_service.get_encounters(curr["id"])

        if not encounters:
            st.info("No saved encounters for this patient yet.")
        else:
            for enc in encounters:
                dt_str = enc.get("timestamp", "Unknown Date")
                summ = enc.get("report_data", {}).get("summary", "No Summary")
                n_flags = len(enc.get("report_data", {}).get("flags", []))

                with st.expander(f"📅 {dt_str} — {n_flags} Flags"):
                    col_h1, col_h2 = st.columns([4, 1])
                    with col_h1:
                        st.caption(summ)
                    with col_h2:
                        if st.button("Restore", key=f"rest_{enc['id']}", help="Load this encounter's data into the editor"):
                            st.session_state.note_in_val = enc.get("input_data", {}).get("note", "")
                            st.session_state.meds_in_val = enc.get("input_data", {}).get("meds", "")
                            st.session_state.labs_in_val = enc.get("input_data", {}).get("labs", "")
                            st.toast("Restored to Editor", icon="⏪")

                    if n_flags > 0:
                        st.markdown("**Flags:**")
                        for f in enc["report_data"]["flags"]:
                            icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}.get(f.get("severity", "MEDIUM"), "⚪")
                            st.markdown(f"- {icon} **{f.get('category','Issue')}**: {f.get('explanation')}")

                    st.markdown("---")
                    st.text("Input Note:")
                    st.text(enc.get("input_data", {}).get("note", ""))

        st.divider()

    st.markdown("#### 🕒 Local Session History (Unsaved Guest Activity)")
    if "review_history" not in st.session_state or not st.session_state.review_history:
        st.info("No session activity yet.")
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
