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
import altair as alt

# Module Imports
import src.domain.models
# importlib.reload(src.domain.models) # Hot-reload schema removed for production stability

from src.adapters.ollama_adapter import ReviewEngineAdapter
from src.core.input_loader import standardize_input
from src.services.audit_service import AuditService
from src.services.image_quality_service import ImageQualityService
from src.services.chat_service import ChatService
from src.domain.models import ChatSession, ChatMessage, PatientRecord, SafetyFlag

from src.eval.run_eval import run_eval_pipeline
from src.services.transcription_service import TranscriptionService
import src.core.extract
importlib.reload(src.core.extract)
from src.core.extract import FactExtractor
import src.services.patient_service
importlib.reload(src.services.patient_service)
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
# Hot-reload check: Re-init if method missing
if "patient_service" not in st.session_state or not hasattr(st.session_state.patient_service, "get_population_stats"):
    st.session_state.patient_service = PatientService()

st.markdown("""
<style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Theme Adaptation: Native Variables */
    :root {
        /* Severity Colors (RGBA for contrast) */
        --sev-high-bg: rgba(239, 68, 68, 0.08);
        --sev-high-border: #ef4444;

        --sev-med-bg: rgba(245, 158, 11, 0.08);
        --sev-med-border: #f59e0b;

        --sev-low-bg: rgba(34, 197, 94, 0.08);
        --sev-low-border: #22c55e;

        --evidence-bg: var(--secondary-background-color);
        --evidence-border: var(--secondary-background-color);
        --evidence-text: var(--text-color);

        /* Badges */
        --badge-note: rgba(59, 130, 246, 0.12);
        --badge-labs: rgba(168, 85, 247, 0.12);
        --badge-meds: rgba(34, 197, 94, 0.12);

        /* Chat */
        --user-bubble-bg: var(--primary-color);
        --user-bubble-text: #ffffff;

        /* Text */
        --text-std: var(--text-color);
        --text-muted: #9ca3af;

        /* Cards */
        --card-bg: var(--secondary-background-color);
        --card-border: rgba(128, 128, 128, 0.15);
        --card-shadow: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
    }

    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .main .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 600; }
    div[data-testid="stMetricLabel"] { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.03em; color: var(--text-muted); }

    /* Severity Block */
    .severity-block {
        padding: 14px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
        color: var(--text-std);
        box-shadow: var(--card-shadow);
    }
    .severity-block h4 {
        margin: 0 0 6px 0;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    .severity-block p {
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .severity-high {
        border-left: 4px solid var(--sev-high-border);
        background-color: var(--sev-high-bg);
    }
    .severity-medium {
        border-left: 4px solid var(--sev-med-border);
        background-color: var(--sev-med-bg);
    }
    .severity-low {
        border-left: 4px solid var(--sev-low-border);
        background-color: var(--sev-low-bg);
    }

    .evidence-block {
        background-color: var(--card-bg);
        border-left: 3px solid var(--text-muted);
        padding: 10px 12px;
        margin: 6px 0;
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 0.85em;
        color: var(--evidence-text);
        border-radius: 0 6px 6px 0;
        line-height: 1.5;
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
        border-radius: 10px;
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        margin-bottom: 20px;
        color: var(--text-std);
        box-shadow: var(--card-shadow);
    }

    .chat-history-box {
        height: 400px;
        overflow-y: auto;
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 15px;
        background-color: var(--card-bg);
        margin-bottom: 10px;
    }

    /* Record cards (read-only view) */
    .record-card {
        background-color: var(--card-bg);
        padding: 16px;
        border-radius: 10px;
        border: 1px solid var(--card-border);
        box-shadow: var(--card-shadow);
        max-height: 400px;
        overflow-y: auto;
        line-height: 1.6;
    }
    .record-card em { color: var(--text-muted); }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 48px 24px;
        background-color: var(--card-bg);
        border-radius: 12px;
        border: 1px dashed var(--card-border);
    }
    .empty-state h3 { color: var(--text-muted); margin-bottom: 8px; font-weight: 600; }
    .empty-state p { color: var(--text-muted); font-size: 0.92rem; }

    /* Advisory banner */
    .advisory-banner {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.08), rgba(245, 158, 11, 0.03));
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.88rem;
        color: var(--text-std);
        margin-bottom: 16px;
    }
    .advisory-banner strong { color: #f59e0b; }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--card-border);
    }
    .sidebar-brand {
        text-align: center;
        padding: 8px 0 4px 0;
    }
    .sidebar-brand h1 {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .sidebar-brand .version {
        display: inline-block;
        background: rgba(139, 92, 246, 0.12);
        color: #8b5cf6;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        margin-top: 4px;
        letter-spacing: 0.02em;
    }

    /* Danger Zone */
    .danger-zone-card {
        background: rgba(239, 68, 68, 0.04);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 10px;
        padding: 16px;
        margin-top: 4px;
    }
    .danger-zone-card .dz-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .danger-zone-card .dz-header .dz-icon {
        font-size: 1.1rem;
    }
    .danger-zone-card .dz-header h4 {
        margin: 0;
        font-size: 0.88rem;
        font-weight: 600;
        color: #ef4444;
    }
    .danger-zone-card .dz-body {
        font-size: 0.84rem;
        color: var(--text-muted);
        line-height: 1.5;
        margin-bottom: 10px;
    }
    .danger-zone-card .dz-body strong {
        color: var(--text-std);
    }

    /* Safety review progress */
    .review-progress {
        padding: 12px 16px;
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        margin-top: 8px;
    }
    .review-progress .step {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
        font-size: 0.88rem;
        color: var(--text-muted);
    }
    .review-progress .step.active {
        color: var(--text-std);
        font-weight: 500;
    }
    .review-progress .step.done {
        color: #22c55e;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h1>🛡️ SentinelMD</h1>
        <p style="color: var(--text-muted); margin: 2px 0 6px 0; font-size: 0.9rem;">Clinical Safety Copilot</p>
        <span class="version">EDGE AI • v1.0</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")


    # Workflow Selection
    input_mode_map = {
        "Patient Records": "Patient Records",
        "Population Health": "Population Health",
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
                with st.sidebar.expander("⚙️ Patient Settings", expanded=False):
                    enc_count = len(st.session_state.patient_service.get_encounters(found_p["id"]))
                    st.markdown(f"""
                    <div class="danger-zone-card">
                        <div class="dz-header">
                            <span class="dz-icon">🗑️</span>
                            <h4>Delete Patient</h4>
                        </div>
                        <div class="dz-body">
                            Remove <strong>{found_p['name']}</strong> and
                            <strong>{enc_count}</strong> encounter{'s' if enc_count != 1 else ''}
                            permanently. This action cannot be undone.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if not st.session_state.get("_confirm_delete"):
                        if st.button("Delete Patient…", key="btn_delete_patient", type="secondary", use_container_width=True):
                            st.session_state._confirm_delete = True
                            st.rerun()
                    else:
                        st.markdown(f'<p style="text-align:center; font-size:0.88rem; color:#ef4444; font-weight:600; margin: 8px 0 4px 0;">⚠️ Are you sure? This is irreversible.</p>', unsafe_allow_html=True)
                        col_del1, col_del2 = st.columns(2)
                        with col_del1:
                            if st.button("Cancel", key="btn_cancel_delete", use_container_width=True):
                                st.session_state._confirm_delete = False
                                st.rerun()
                        with col_del2:
                            if st.button("🗑️ Confirm Delete", key="btn_confirm_delete", type="primary", use_container_width=True):
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
        selected_model = "amsaravi/medgemma-4b-it:q6"
        st.session_state.backend_type = "ollama"
        st.markdown("**Inference Engine**")
        st.caption("🧠 MedGemma 4B (Local, Quantized)")

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
st.markdown('<div class="advisory-banner">⚠️ <strong>ADVISORY ONLY</strong> : Identifying safety risks. Not a substitute for clinical judgment.</div>', unsafe_allow_html=True)

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
                 <div class="empty-state">
                    <h3>📭 Patient Record is Empty</h3>
                    <p>No clinical data has been recorded yet.</p>
                    <p>Click the <b>✏️ Edit Record</b> button above to start documenting.</p>
                 </div>
                 """, unsafe_allow_html=True)
            else:
                 c1, c2, c3 = st.columns(3)

                 # Helper style
                 box_style = "record-card"

                 with c1:
                     st.markdown("### 📝 Clinical Note")
                     content = note_in if note_in else '<em>No note recorded.</em>'
                     st.markdown(f'<div class="{box_style}">{content}</div>', unsafe_allow_html=True)
                 with c2:
                     st.markdown("### 🧪 Labs")
                     content = labs_in if labs_in else '<em>No labs recorded.</em>'
                     st.markdown(f'<div class="{box_style}" style="white-space: pre-wrap;">{content}</div>', unsafe_allow_html=True)
                 with c3:
                     st.markdown("### 💊 Medications")
                     content = meds_in if meds_in else '<em>No medications recorded.</em>'
                     st.markdown(f'<div class="{box_style}" style="white-space: pre-wrap;">{content}</div>', unsafe_allow_html=True)


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

elif input_mode == "Population Health":
    st.header("Population Health Analytics")
    st.caption("Aggregate safety insights across your patient panel.")

    stats = st.session_state.patient_service.get_population_stats()

    # 1. Key Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Patients", stats["total_patients"])
    m2.metric("High Risk", stats["risk_distribution"]["High"], delta_color="inverse")
    m3.metric("Medium Risk", stats["risk_distribution"]["Medium"], delta_color="off")
    m4.metric("Active Safety Flags", sum(stats["top_flags"].values()))

    st.divider()

    # 2. Charts
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk Stratification")
        risk_data = pd.DataFrame({
            "Risk Level": list(stats["risk_distribution"].keys()),
            "Patients": list(stats["risk_distribution"].values())
        })
        # Custom sort order
        risk_order = ["High", "Medium", "Low", "Unknown"]
        chart_risk = alt.Chart(risk_data).mark_bar().encode(
            x=alt.X('Risk Level', sort=risk_order),
            y='Patients',
            color=alt.Color('Risk Level', scale=alt.Scale(domain=['High', 'Medium', 'Low', 'Unknown'], range=['#ef4444', '#f59e0b', '#22c55e', '#94a3b8'])),
            tooltip=['Risk Level', 'Patients']
        ).properties(height=300)
        st.altair_chart(chart_risk, use_container_width=True)

    with c2:
        st.subheader("Top Safety Concerns")
        if stats["top_flags"]:
            flag_data = pd.DataFrame({
                "Category": list(stats["top_flags"].keys()),
                "Count": list(stats["top_flags"].values())
            })
            chart_flags = alt.Chart(flag_data).mark_bar().encode(
                x='Count',
                y=alt.Y('Category', sort='-x'),
                color=alt.value('#6366f1'),
                tooltip=['Category', 'Count']
            ).properties(height=300)
            st.altair_chart(chart_flags, use_container_width=True)
        else:
            st.info("No safety flags detected yet.")

    # 3. Patient List with Risk
    st.subheader("Patient Panel")
    all_p = st.session_state.patient_service.get_all_patients()

    # Enrich with risk
    table_data = []
    for p in all_p:
        encs = st.session_state.patient_service.get_encounters(p["id"])
        risk = "Unknown"
        last_seen = "Never"
        flags = 0

        if encs:
            last_seen = encs[0].get("timestamp", "")[:10]
            report = encs[0].get("report", {})
            r_flags = report.get("flags", [])
            flags = len(r_flags)

            if not r_flags:
                risk = "Low"
            else:
                max_sev = "Low"
                for f in r_flags:
                    sev = f.get("severity", "LOW").upper()
                    # Handle "SafetySeverity.HIGH" or "HIGH"
                    if "HIGH" in sev:
                        max_sev = "High"
                        break
                    elif "MEDIUM" in sev and max_sev != "High":
                        max_sev = "Medium"
                risk = max_sev

        table_data.append({
            "Name": p["name"],
            "DOB": p["dob"],
            "MRN": p.get("mrn", ""),
            "Risk Status": risk,
            "Active Flags": flags,
            "Last Audit": last_seen
        })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            column_config={
                "Risk Status": st.column_config.TextColumn(
                    "Risk Status",
                    help="Highest severity flag in latest audit",
                    validate="^(High|Medium|Low|Unknown)$"
                ),
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No patients found.")


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

                    # --- NEW: "Voice-to-Chart" Auto-Extraction ---
                    with st.spinner("✨ Extraction: Parsing Meds & Labs from dictation..."):
                        if "fact_extractor" not in st.session_state:
                            model_name = os.getenv("OLLAMA_MODEL", "amsaravi/medgemma-4b-it:q6")
                            st.session_state.fact_extractor = FactExtractor(
                                backend_type="ollama",
                                backend_url="http://localhost:11434",
                                model=model_name
                            )

                        parsed = st.session_state.fact_extractor.parse_dictation(text)

                        # Update session state with parsed values
                        st.session_state.note_in_val = parsed.get("note_section", text)
                        st.session_state.meds_in_val = parsed.get("medications", "")
                        st.session_state.labs_in_val = parsed.get("labs", "")

                        # Sync Widgets to ensure UI updates
                        st.session_state.widget_paste_note_struct = st.session_state.note_in_val
                        st.session_state.widget_paste_meds = st.session_state.meds_in_val
                        st.session_state.widget_paste_labs = st.session_state.labs_in_val

                        # Set a flag to show the AI badge
                        st.session_state.voice_parsed_success = True

                    n_meds = len(st.session_state.meds_in_val.splitlines()) if st.session_state.meds_in_val else 0
                    n_labs = len(st.session_state.labs_in_val.splitlines()) if st.session_state.labs_in_val else 0

                    st.toast(f"✅ Voice-to-Chart Complete! Found {n_meds} meds, {n_labs} labs.", icon="🪄")
                    st.rerun()  # Force clean redraw to prevent UI ghosting

                except Exception as e:
                    st.error(f"Processing Error: {e}")

    # Sync Text Area with Session State
    if "note_in_val" not in st.session_state: st.session_state.note_in_val = ""
    if "meds_in_val" not in st.session_state: st.session_state.meds_in_val = ""
    if "labs_in_val" not in st.session_state: st.session_state.labs_in_val = ""

    # Check if voice was used (transcription exists)
    # Init widget defaults from session state
    if "widget_paste_note_struct" not in st.session_state: st.session_state.widget_paste_note_struct = st.session_state.get("note_in_val", "")
    if "widget_paste_meds" not in st.session_state: st.session_state.widget_paste_meds = st.session_state.get("meds_in_val", "")
    if "widget_paste_labs" not in st.session_state: st.session_state.widget_paste_labs = st.session_state.get("labs_in_val", "")

    # Check for "Voice-to-Chart" success
    if st.session_state.get("voice_parsed_success", False):
        st.info("✨ **AI Scribe Active**: Medications and Labs have been automatically extracted from your dictation.", icon="🪄")

    # Structured input view (Always visible now)
    with col_p1:
            note_in = st.text_area("Clinical Note", height=300,
                                placeholder="Example:\nPt presented with weakness...\nPMH: CKD, HTN...",
                                key="widget_paste_note_struct",
                                help="Narrative section extracted from dictation")
            st.session_state.note_in_val = note_in

    with col_p2:
            labs_in = st.text_area("Labs", height=300,
                                placeholder="Example:\nPotassium: 6.0 mmol/L\nCreatinine: 2.1 mg/dL",
                                key="widget_paste_labs",
                                help="Auto-extracted labs")
            st.session_state.labs_in_val = labs_in

    with col_p3:
            meds_in = st.text_area("Medications", height=300,
                                placeholder="Example:\nLisinopril 10mg daily\nSpironolactone 25mg daily",
                                key="widget_paste_meds",
                                help="Auto-extracted medications")
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
    run_btn = st.button("🛡️  Run Safety Review", type="primary", use_container_width=False, key="btn_run_safety")

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
                st.markdown("""
                <div style="display:flex; align-items:center; gap:10px; padding:10px 16px;
                            background: rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.25);
                            border-radius:8px; margin:8px 0;">
                    <span style="font-size:1.2rem;">⚡</span>
                    <span style="font-size:0.9rem; color:var(--text-std); font-weight:500;">
                        Analysis loaded from cache — instant result
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.status("🛡️ Running Safety Review…", expanded=True) as status:
                    t0 = time.time()

                    st_val = status.empty()
                    st_val.write("⏳ Validating clinical inputs...")
                    note_text = standardized_inputs["note_text"]
                    labs_text = standardized_inputs["labs_text"]
                    meds_text = standardized_inputs["meds_text"]
                    time.sleep(0.5) # Slight pause for UX
                    st_val.write("✅ Validating clinical inputs... Done")

                    # DDI pre-scan (deterministic, instant)
                    st_ddi = status.empty()
                    st_ddi.write("⏳ Running DDI pre-scan...")
                    from src.core.ddi_checker import extract_medications, check_interactions
                    parsed_meds = extract_medications(meds_text)
                    ddi_hits = check_interactions(parsed_meds)
                    n_meds = len(parsed_meds)
                    n_ddi = len(ddi_hits)

                    if n_ddi > 0:
                        st_ddi.write(f"💊 DDI pre-scan: {n_meds} medications → **{n_ddi} interaction{'s' if n_ddi != 1 else ''} detected**")
                    else:
                        st_ddi.write(f"💊 DDI pre-scan: {n_meds} medications — no known interactions")

                    st_llm = status.empty()
                    st_llm.write("🧠 Analyzing with MedGemma 4B (on-device)...")
                    report = st.session_state.audit_service.run_safety_review(
                        note_text, labs_text, meds_text
                    )
                    st_llm.write("🧠 Analysis complete.")

                    elapsed = time.time() - t0
                    status.write(f"📋 Report generated — {elapsed:.1f}s")

                    n_flags = len(report.flags) if report and hasattr(report, 'flags') else 0
                    status.update(
                        label=f"✅ Analysis complete · {n_flags} flag{'s' if n_flags != 1 else ''} · {elapsed:.1f}s",
                        state="complete",
                        expanded=False
                    )

                if report:
                    st.session_state.last_report = report
                    st.session_state.inference_cache[cache_key] = {"report": report}

                    # --- AUTO-DETECT PATIENT ---
                    if report.patient_demographics and "name" in report.patient_demographics:
                        detected_name = report.patient_demographics["name"]
                        detected_dob = report.patient_demographics.get("dob", "Unknown")

                        current_p = st.session_state.get("current_patient")

                        if not current_p:
                            all_p = st.session_state.patient_service.get_all_patients()
                            match = None
                            for p in all_p:
                                if p["name"].lower() == detected_name.lower():
                                    match = p
                                    break

                            if match:
                                st.session_state.current_patient = match
                                st.toast(f"Matched existing patient: {detected_name}", icon="🔗")
                            else:
                                new_p = st.session_state.patient_service.create_patient(detected_name, detected_dob)
                                if new_p is None:
                                    st.warning(f"Patient '{detected_name}' already exists - skipping auto-create")
                                else:
                                    st.session_state.current_patient = new_p
                                    st.toast(f"Auto-created patient: {detected_name}", icon="✨")

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

                    st.rerun()  # Force clean redraw
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
                    display_cat = "Drug-Drug Interaction"

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
            from datetime import datetime as _dt

            for enc in encounters:
                dt_raw = enc.get("timestamp", "Unknown Date")
                # Format timestamp nicely
                try:
                    dt_obj = _dt.fromisoformat(str(dt_raw))
                    dt_str = dt_obj.strftime("%b %d, %Y at %I:%M %p")
                except (ValueError, TypeError):
                    dt_str = str(dt_raw)

                report_data = enc.get("report_data", {})
                summ = report_data.get("summary", "")
                flags = report_data.get("flags", [])
                n_flags = len(flags)
                input_data = enc.get("input_data", {})
                has_audit = bool(summ or n_flags > 0)

                # Build expander label
                if has_audit:
                    sev_icon = "🔴" if any(f.get("severity") == "HIGH" for f in flags) else ("🟠" if n_flags > 0 else "🟢")
                    label = f"{sev_icon} {dt_str} — {n_flags} Flag{'s' if n_flags != 1 else ''}"
                else:
                    # Unaudited encounter — show note preview
                    note_preview = input_data.get("note", "")[:60]
                    if len(input_data.get("note", "")) > 60:
                        note_preview += "..."
                    label = f"📋 {dt_str} — Recorded (not yet audited)"

                with st.expander(label, expanded=False):
                    # Restore button + summary row
                    col_h1, col_h2 = st.columns([4, 1])
                    with col_h1:
                        if has_audit:
                            st.caption(f"**Analysis:** {summ}")
                        else:
                            st.caption("This encounter has clinical data but hasn't been audited yet.")
                    with col_h2:
                        if st.button("⏪ Restore", key=f"rest_{enc['id']}", help="Load this encounter's data into the editor"):
                            st.session_state.note_in_val = input_data.get("note", "")
                            st.session_state.meds_in_val = input_data.get("meds", "")
                            st.session_state.labs_in_val = input_data.get("labs", "")
                            st.toast("Restored to Editor", icon="⏪")

                    if n_flags > 0:
                        st.markdown("**Flags:**")
                        for f in flags:
                            icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}.get(f.get("severity", "MEDIUM"), "⚪")
                            cat = f.get('category', 'Issue').replace('_', ' ').title()
                            st.markdown(f"- {icon} **[{f.get('severity', 'MEDIUM')}] {cat}**: {f.get('explanation')}")

                    # Show input preview
                    note_text = input_data.get("note", "")
                    if note_text:
                        with st.expander("📝 View Clinical Note", expanded=False):
                            st.markdown(f'<div class="record-card" style="white-space: pre-wrap; font-size: 0.9rem;">{note_text}</div>', unsafe_allow_html=True)

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
    """

    # --- CSS STYLES ---
    st.markdown("""
    <style>
    /* Floating Action Button */
    div[data-testid="stPopover"] {
        position: fixed !important;
        bottom: 28px !important;
        right: 28px !important;
        width: 56px !important;
        height: 56px !important;
        z-index: 9999 !important;
        background-color: transparent !important;
    }

    div[data-testid="stPopover"] > button {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
        color: white !important;
        border-radius: 50% !important;
        width: 56px !important;
        height: 56px !important;
        box-shadow: 0 4px 14px rgba(109, 40, 217, 0.35) !important;
        border: none !important;
        font-size: 24px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }

    div[data-testid="stPopover"] > button:hover {
        transform: scale(1.08) !important;
        box-shadow: 0 6px 20px rgba(109, 40, 217, 0.5) !important;
    }

    div[data-testid="stPopover"] > button > div {
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Chat Bubbles — theme-safe */
    .chat-bubble-user {
        background: linear-gradient(135deg, #7c3aed, #6d28d9);
        color: #fff;
        padding: 10px 14px;
        border-radius: 16px 16px 4px 16px;
        margin-bottom: 10px;
        font-size: 0.88rem;
        line-height: 1.55;
        max-width: 82%;
        float: right;
        clear: both;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    .chat-bubble-bot {
        background: var(--secondary-background-color);
        color: var(--text-color);
        padding: 10px 14px;
        border-radius: 16px 16px 16px 4px;
        margin-bottom: 10px;
        font-size: 0.88rem;
        line-height: 1.55;
        max-width: 82%;
        float: left;
        clear: both;
        border: 1px solid var(--card-border, rgba(128,128,128,0.12));
    }

    /* Chat Header */
    .chat-header {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
        padding: 14px 16px;
        border-radius: 8px 8px 0 0;
        color: white;
        margin: -1rem -1rem 0.75rem -1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .chat-header-avatar {
        font-size: 20px;
        background: rgba(255,255,255,0.15);
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .chat-header-info h4 {
        margin: 0;
        color: white;
        font-size: 0.92rem;
        font-weight: 600;
    }
    .chat-header-info p {
        margin: 0;
        color: rgba(255,255,255,0.7);
        font-size: 0.75rem;
    }

    /* Empty chat state */
    .chat-empty {
        text-align: center;
        padding: 32px 16px;
        color: var(--text-muted, #9ca3af);
    }
    .chat-empty .chat-empty-icon {
        font-size: 2rem;
        margin-bottom: 8px;
        opacity: 0.6;
    }
    .chat-empty h5 {
        margin: 0 0 4px 0;
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--text-color);
    }
    .chat-empty p {
        margin: 0;
        font-size: 0.82rem;
        line-height: 1.5;
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
                <p>Ask about flags, evidence, or compliance</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- CHAT LOGIC ---
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = ChatSession()

        report = st.session_state.get("last_report")

        if not report:
            st.markdown("""
            <div class="chat-empty">
                <div class="chat-empty-icon">🛡️</div>
                <h5>No analysis loaded</h5>
                <p>Run a <b>Safety Review</b> first,<br>then ask me about the results.</p>
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
            chat_cont = st.container(height=380)
            with chat_cont:
                for msg in st.session_state.chat_session.history:
                    if msg.role == "user":
                        st.markdown(f'<div class="chat-bubble-user">{msg.content}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-bubble-bot">{msg.content}</div>', unsafe_allow_html=True)

                st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)

            # Input Area
            if query := st.chat_input("Ask about this review…", key="float_chat_premium"):
                st.session_state.chat_session.history.append(ChatMessage(role="user", content=query))
                st.rerun()

            # Generation
            if st.session_state.chat_session.history and st.session_state.chat_session.history[-1].role == "user":
                with chat_cont:
                    with st.spinner("Thinking…"):
                        reply = st.session_state.chat_service.generate_reply(
                            st.session_state.chat_session,
                            st.session_state.chat_session.history[-1].content
                        )
                        st.session_state.chat_session.history.append(ChatMessage(role="assistant", content=reply))
                        st.rerun()

# Call the function
render_floating_chat(standardized_inputs)
