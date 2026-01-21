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
importlib.reload(src.domain.models) # Hot-reload schema

from src.adapters.ollama_adapter import ReviewEngineAdapter
from src.core.input_loader import standardize_input
from src.services.audit_service import AuditService
from src.services.image_quality_service import ImageQualityService
from src.services.chat_service import ChatService
from src.domain.models import ChatSession, ChatMessage, PatientRecord, SafetyFlag
from src.eval.run_eval import run_eval_pipeline

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

    # Input Mode
    input_mode = st.radio(
        "Input Mode",
        ["Demo Cases", "Paste Text", "Upload Files"],
        help="Select input method."
    )

    st.markdown("---")

    # Demo Selector
    selected_case_file = None
    if input_mode == "Demo Cases":
        DATA_DIR = "data/synthetic"
        cases = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])
        selected_case_file = st.selectbox("Select Case", cases, index=0)

    run_btn = st.button("Run Safety Review", type="primary", use_container_width=True)

    # System Status
    st.markdown("---")
    with st.expander("🔧 System Status", expanded=True):
        # 1. Review Engine
        model_map = {
            "MedGemma 4B": "amsaravi/medgemma-4b-it:q6",
            "Mock Model (Test)": "mock-model"
        }

        display_name = st.selectbox(
            "Local Review Engine",
            options=list(model_map.keys()),
            index=0,
            help="Select local inference model."
        )
        selected_model = model_map[display_name]
        st.session_state.backend_type = "ollama"

        # Init Services
        if "audit_service" not in st.session_state or st.session_state.current_model != selected_model:
            with st.spinner("Initializing Engine..."):
                try:
                    adapter = ReviewEngineAdapter(model=selected_model)

                    if not adapter.check_connection():
                         st.error("⚠️ Engine Offline. Run `ollama serve`.")
                    else:
                         st.success(f"✅ Connected: {selected_model}")

                    st.session_state.audit_service = AuditService(adapter)
                    st.session_state.chat_service = ChatService(adapter)
                    st.session_state.current_model = selected_model
                    st.session_state.inference_cache = {}

                except Exception as e:
                    st.error(f"Init Failed: {e}")

        # 2. Tech Specs
        st.markdown(f"""
        <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
        • <b>OCR Engine:</b> Local (Tesseract)<br>
        • <b>Privacy:</b> 100% Offline<br>
        • <b>Visual Review:</b> Deterministic Only
        </div>
        """, unsafe_allow_html=True)

    # Options
    one_call_mode = st.checkbox("🎯 One-Call Demo Mode", value=False, help="Fast rule-based extraction.")
    st.session_state.one_call_mode = one_call_mode

    st.session_state.max_flags = 10
    st.session_state.extract_opts = {"num_ctx": 4096, "num_predict": 350, "temperature": 0.0}
    st.session_state.audit_opts = {"num_ctx": 4096, "num_predict": 450, "temperature": 0.0}

    with st.expander("⚙️ Settings"):
        st.caption(f"**Extract**: ctx={st.session_state.extract_opts['num_ctx']}, tokens={st.session_state.extract_opts['num_predict']}")
        st.caption(f"**Audit**: ctx={st.session_state.audit_opts['num_ctx']}, tokens={st.session_state.audit_opts['num_predict']}")
        st.caption(f"**Max Flags**: {st.session_state.max_flags}")

    with st.expander("⏱️ Metrics", expanded=True):
        if "last_report" in st.session_state:
            report = st.session_state.last_report
            ext_time = report.metadata.get("extract_runtime", 0)
            aud_time = report.metadata.get("audit_runtime", 0)
            total_time = report.metadata.get("total_runtime", ext_time + aud_time)

            st.markdown(f"**Total**: `{total_time:.2f}s`")
            st.text(f"Extract: {ext_time:.2f}s")
            st.text(f"Audit:   {aud_time:.2f}s")
        else:
            st.caption("No metrics available.")

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
    st.header(f"Mode: Demo Case ({selected_case_file})")
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
    st.header("Mode: Paste Clinical Text")
    st.caption("Enter clinical data below.")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        note_in = st.text_area("Clinical Note", height=300, placeholder="Example:\nPt presented with weakness...\nPMH: CKD, HTN...")
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
    st.header("Mode: Upload Files")
    st.caption("Supports TXT, PDF, CSV. Multiple files merged.")

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

# Tabs
tab_safety, tab_inputs, tab_eval = st.tabs(["🛡️ Safety Review", "📄 Clinical Inputs", "📊 Evaluation"])

# Tab 1: Safety Review
with tab_safety:
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
                with st.spinner("Running Safety Review (Local Engine)..."):
                    t0 = time.time()

                    note_text = standardized_inputs["note_text"]
                    labs_text = standardized_inputs["labs_text"]
                    meds_text = standardized_inputs["meds_text"]

                    # Audit Execution
                    report = st.session_state.audit_service.run_safety_review(
                        note_text, labs_text, meds_text
                    )

                    if report:
                        st.session_state.last_report = report
                        st.session_state.inference_cache[cache_key] = {"report": report}
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

        st.markdown(f"""
        <div class="summary-card">
            <h3>Audit Summary</h3>
            <p><b>{flag_count}</b> Safety Flag(s) Identified</p>
            <p>Highest Severity: <b>{max_severity}</b></p>
            <p style="color: var(--text-std); font-size: 0.9em;">{report.summary}</p>
        </div>
        """, unsafe_allow_html=True)



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
                    st.markdown(f"""
                    <div class="{css_class} severity-block">
                        <h4>[{sev_val}] {display_cat}</h4>
                        <p><b>{flag.explanation}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

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

                    with st.expander("🔍 Click for Detailed Logic Breakdown"):
                        # Get Primary Evidence Details
                        ev_source = flag.evidence[0].source if flag.evidence else "General"
                        ev_quote = flag.evidence[0].quote if flag.evidence else "Entire Case Context"

                        st.markdown(f"""
                        **1. Evidence Found**
                        - **Source**: `{ev_source}`
                        - **Content**: *"{ev_quote}"*

                        **2. Clinical Reasoning**
                        {flag.explanation}

                        **3. Risk Analysis**
                        - **Category**: {display_cat}
                        - **Severity**: {sev_val}

                        **4. Recommendation**
                        {flag.recommendation if flag.recommendation else "Review clinical guidelines."}
                        """)

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
        st.markdown("### 💬 Safety Review Assistant")

        if "chat_session" not in st.session_state:
            st.session_state.chat_session = ChatSession()

        # Backward Compatibility: Ensure suggested_replies exists if session is stale
        if not hasattr(st.session_state.chat_session, "suggested_replies"):
            st.session_state.chat_session.suggested_replies = []

        # Reset session if audit changed (only if run_btn was pressed recently)
        # Note: We rely on the fact that if report exists, we can chat.

        # State Management: Init
        if not report:
             st.info("Run a Safety Review to enable the assistant.")
        else:
            # 1. State Update
            audit_dict = report.model_dump()
            # Create meaningful context for the assistant
            raw_note = standardized_inputs.get('note_text', '')
            # Take safe header/context (first 2k chars) to capture demographics + HPI
            context_snippet = raw_note[:2000] + ("..." if len(raw_note) > 2000 else "")
            input_summary_txt = f"Clinical Note Content:\n{context_snippet}"

            st.session_state.chat_session = st.session_state.chat_service.reset_session(
                st.session_state.chat_session,
                audit_dict,
                input_summary_txt
            )

            # 2. Chat Layout (Height constrained container)
            chat_container = st.container(height=500)

            with chat_container:
                # Render History
                for msg in st.session_state.chat_session.history:
                    if msg.role == "user":
                        # Custom User Bubble (Right Aligned)
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-end; align-items: flex-start; margin: 10px 0;">
                            <div style="background-color: #007bff; color: white; padding: 10px 15px; border-radius: 15px 15px 0 15px; margin-right: 10px;">
                                {msg.content}
                            </div>
                            <div style="font-size: 24px; margin-top: -5px;">🧑‍⚕️</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Standard Assistant Bubble (Left)
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-start; align-items: flex-start; margin: 10px 0;">
                            <div style="font-size: 24px; margin-top: -5px; margin-right: 10px;">🛡️</div>
                            <div style="background-color: rgba(128,128,128,0.1); padding: 10px 15px; border-radius: 0 15px 15px 15px; color: var(--text-std);">
                                {msg.content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # 3. Suggested Replies (Chips)
            # Only show if no pending input and we have suggestions
            query = None
            if st.session_state.chat_session.suggested_replies:
                st.write("") # Spacer
                cols = st.columns(len(st.session_state.chat_session.suggested_replies))
                for i, chip_text in enumerate(st.session_state.chat_session.suggested_replies):
                    if cols[i].button(chip_text, key=f"chip_{len(st.session_state.chat_session.history)}_{i}"):
                        query = chip_text

            # 4. Chat Input
            if user_input := st.chat_input("Ask about safety flags..."):
                query = user_input

            # 5. Execution Loop
            if query:
                # Add User Message
                st.session_state.chat_session.history.append(ChatMessage(role="user", content=query))
                st.session_state.chat_session.suggested_replies = [] # Clear chips
                st.rerun()

            # Handle Response Generation (post-rerun)
            if st.session_state.chat_session.history and st.session_state.chat_session.history[-1].role == "user":
                with chat_container:

                     with st.chat_message("assistant", avatar="🛡️"):
                        # Visual "thinking" indicator
                        think_box = st.empty()
                        think_box.markdown("...")

                        # Pre-compute response (blocking)
                        full_response = st.session_state.chat_service.generate_reply(
                            st.session_state.chat_session,
                            st.session_state.chat_session.history[-1].content
                        )

                        # Clear indicator
                        think_box.empty()

                        # Stream response
                        def fast_stream():
                            import time
                            words = full_response.split()
                            for w in words:
                                yield w + " "
                                time.sleep(0.02)

                        response_stream = st.write_stream(fast_stream)

                        # Save to history
                        st.session_state.chat_session.history.append(ChatMessage(role="assistant", content=response_stream))

                        # Regenerate chips
                        st.session_state.chat_session.suggested_replies = st.session_state.chat_service.generate_suggestions(
                            st.session_state.chat_session.context, response_stream
                        )
                        st.rerun()

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
        <div style="height: 500px; overflow-y: auto; background-color: #f0f2f6; color: #31333F; padding: 10px; border-radius: 5px; font-family: monospace; white-space: pre-wrap;">{display_content if display_content else "No clinical note provided."}</div>
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
            # Text View
            st.info("Displaying parsed input text used for analysis (Structured view not available for raw input).")

            st.markdown("**Medications (Text)**")
            st.text_area("Meds", standardized_inputs["meds_text"], height=150, disabled=True)

            st.markdown("**Laboratories (Text)**")
            st.text_area("Labs", standardized_inputs["labs_text"], height=200, disabled=True)

# Tab 3: Evaluation
with tab_eval:
    st.subheader("System Evaluation")

    if input_mode != "Demo Cases":
        st.info("ℹ️ Evaluation metrics are available only for **Demo Cases** because they contain ground truth labels.\n\nPlease switch to 'Demo Cases' mode to run the evaluation suite.")
    else:
        st.caption("Run a full evaluation against the synthetic dataset.")

    if st.button("▶️ Run Evaluation Suite"):
        with st.spinner("Running batch evaluation against 'data/synthetic/'..."):
            run_eval_pipeline()
            st.success("Evaluation complete! Results updated.")
            st.rerun()

    results_path = "results.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            try: results = json.load(f)
            except: results = {}

        summary = results.get("summary", {})
        st.markdown("### Aggregate Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("F1 Score", f"{summary.get('f1', 0):.2f}")
        m2.metric("Weighted Recall", f"{summary.get('weighted_recall', 0):.2f}")
        m3.metric("High-Sev Recall", f"{summary.get('high_severity_recall', 0):.2f}")
        m4.metric("Avg FDR", f"{summary.get('avg_fpr_fdr', 0):.2f}")

        st.caption(f"Evidence Grounding: {summary.get('grounding_rate', 0):.1%} | Avg Runtime: {summary.get('avg_runtime_sec', 0):.2f}s")
        st.divider()

        # Cases Table
        if results.get("cases"):
            df_data = []
            for c in results["cases"]:
                m = c["metrics"]
                df_data.append({
                    "Case": c["filename"],
                    "TP": c.get("tp", 0),
                    "FP": c.get("fp", 0),
                    "FN": c.get("fn", 0),
                    "W.Recall": m.get("weighted_recall", 0),
                    "Grounding": m.get("grounding_rate", 0),
                    "Runtime (s)": c.get("duration_sec", 0)
                })
            st.dataframe(pd.DataFrame(df_data), use_container_width=True, hide_index=True)
    else:
            st.info("No results found. Click 'Run Evaluation Suite' to generate.")
