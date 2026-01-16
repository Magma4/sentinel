import streamlit as st
import json
import os
import sys
import time
import hashlib

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from typing import Dict, Any, Optional
from src.core.audit import SafetyAuditor
from src.core.extract import FactExtractor
from src.core.schema import PatientRecord, SafetyFlag
from src.eval.run_eval import run_eval_pipeline
from src.core.gating import gate_safety_flags, calibrate_confidence

# Force reload of core modules to ensure updates apply immediately
import importlib
import src.core.input_loader
import src.core.vision_client
import src.core.audit
importlib.reload(src.core.input_loader)
importlib.reload(src.core.vision_client)
importlib.reload(src.core.audit)
from src.core.input_loader import standardize_input
from src.core.pdf_utils import analyze_pdf_quality

# --- Configuration & Styling ---
st.set_page_config(
    page_title="SentinelMD",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
    .severity-high { border-left: 5px solid #ff4b4b; padding-left: 10px; background-color: #fff1f0; padding: 10px; border-radius: 0 5px 5px 0; margin-bottom: 10px;}
    .severity-medium { border-left: 5px solid #ffa421; padding-left: 10px; background-color: #fff8eb; padding: 10px; border-radius: 0 5px 5px 0; margin-bottom: 10px;}
    .severity-low { border-left: 5px solid #21c354; padding-left: 10px; background-color: #f6fffa; padding: 10px; border-radius: 0 5px 5px 0; margin-bottom: 10px;}
    .evidence-box { background-color: #f0f2f6; padding: 5px 10px; border-radius: 4px; font-family: monospace; font-size: 0.9em; margin-top: 5px; }
    .summary-card { padding: 15px; border-radius: 8px; background-color: #f8f9fa; border: 1px solid #e9ecef; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("🛡️ SentinelMD")
    st.caption("Clinical Safety Copilot")
    st.markdown("---")

    # Input Mode Selector
    input_mode = st.radio(
        "Input Mode",
        ["Demo Cases", "Paste Text", "Upload Files"],
        help="Select how you want to provide clinical data."
    )

    st.markdown("---")

    # Case Selector (only for Demo)
    selected_case_file = None
    if input_mode == "Demo Cases":
        DATA_DIR = "data/synthetic"
        cases = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])
        selected_case_file = st.selectbox("Select Case", cases, index=0)

    run_btn = st.button("Run Safety Review", type="primary", use_container_width=True)

    # --- Sidebar: Controls ---
    st.markdown("---")
    st.header("Runtime Environment")

    # LLM Selection
    AVAILABLE_MODELS = [
        {"name": "medgemma:2b", "base_url": "http://localhost:11434"},
        {"name": "gemma2:9b", "base_url": "http://localhost:11434"}, # Balanced (Best for 16GB RAM)
        {"name": "amsaravi/medgemma-4b-it:q6", "base_url": "http://localhost:11434"}, # MedGemma 4B (Multimodal capable)
        {"name": "gemma2:27b", "base_url": "http://localhost:11434"}, # High Performance (Warning: Slow on <32GB RAM)
        {"name": "mock-model", "base_url": None} # For offline testing
    ]
    models = [m["name"] for m in AVAILABLE_MODELS]
    selected_model_name = st.selectbox("LLM Backend (Reasoning)", models, index=0, help="Select the local LLM to run the audit.")

    # Vision Specialist (Multimodal)
    # We prioritize 'paligemma' (Google HAI-DEF) running via Local Transformers (Judge-Safe)
    vision_models = ["paligemma", "moondream", "llava", "medgemma"]
    selected_vision = st.selectbox(
        "Vision Model (Images)",
        vision_models,
        index=0,
        help="Specialized model for X-rays. 'paligemma' runs NATIVELY via Transformers (offline, MPS-accelerated)."
    )

    backend_type = "ollama"

    # Initialize Auditor
    # Re-init if model OR vision model changes
    current_vision = st.session_state.get("current_vision", None)

    if "auditor" not in st.session_state or st.session_state.auditor.model_metadata["name"] != selected_model_name or current_vision != selected_vision:
            model_meta = next(m for m in AVAILABLE_MODELS if m["name"] == selected_model_name)

            if model_meta["name"] == "mock-model":
                backend_type = "mock"
            else:
                backend_type = "ollama"

            st.session_state.auditor = SafetyAuditor(model_meta["base_url"], backend_type, vision_model=selected_vision)
            st.session_state.extractor = FactExtractor(model_meta["base_url"], backend_type)
            st.session_state.backend_type = backend_type
            st.session_state.current_vision = selected_vision
            st.session_state.inference_cache = {}

    # Connection Status
    if st.session_state.backend_type == "ollama":
        st.success(f"✅ **Connected via Ollama**\n\nEndpoint: `{st.session_state.auditor.backend_url}`\n\n**Privacy Check**: Data is processed locally.")
    else:
        st.success("✅ **Offline Mock Mode**\n\nRunning rigid rules. No LLM used.\n\n**Privacy Check**: No data leaves this session.")

    st.markdown("---")

    # Options
    one_call_mode = st.checkbox("🎯 One-Call Demo Mode", value=False, help="Use rule-based extraction for speed.")
    st.session_state.one_call_mode = one_call_mode

    st.session_state.max_flags = 10
    st.session_state.extract_opts = {"num_ctx": 4096, "num_predict": 350, "temperature": 0.0}
    st.session_state.audit_opts = {"num_ctx": 4096, "num_predict": 450, "temperature": 0.0}

    with st.expander("⚙️ Active Settings"):
        st.caption(f"**Extract**: ctx={st.session_state.extract_opts['num_ctx']}, tokens={st.session_state.extract_opts['num_predict']}")
        st.caption(f"**Audit**: ctx={st.session_state.audit_opts['num_ctx']}, tokens={st.session_state.audit_opts['num_predict']}")
        st.caption(f"**Max Flags**: {st.session_state.max_flags}")

    with st.expander("⏱️ Runtime Performance", expanded=True):
        if "last_report" in st.session_state:
            report = st.session_state.last_report
            ext_time = report.metadata.get("extract_runtime", 0)
            aud_time = report.metadata.get("audit_runtime", 0)
            total_time = report.metadata.get("total_runtime", ext_time + aud_time)

            st.markdown(f"**Total**: `{total_time:.2f}s`")
            st.text(f"Extract: {ext_time:.2f}s")
            st.text(f"Audit:   {aud_time:.2f}s")
        else:
            st.caption("Run a review to see metrics.")

# --- Main Content ---
st.warning("⚠️ **ADVISORY ONLY**: This tool helps identify safety risks but does not replace clinical judgment. Clinician retains full responsibility for diagnosis and treatment.")

# Initialize centralized input variables
record: Optional[PatientRecord] = None
standardized_inputs: Dict[str, str] = {
    "case_id": "UNKNOWN",
    "note_text": "",
    "labs_text": "",
    "meds_text": ""
}
file_source_type = "TEXT" # For UI display
quality_report = None # For PDF quality

# --- Input Handling Logic ---
if input_mode == "Demo Cases":
    file_source_type = "DEMO"
    st.header(f"Mode: Demo Case ({selected_case_file})")
    current_case_path = os.path.join(DATA_DIR, selected_case_file)
    with open(current_case_path, "r") as f:
        case_data = json.load(f)
        record = PatientRecord(**case_data)

    # Populate inputs from record
    standardized_inputs = {
        "case_id": selected_case_file.replace(".json", ""),
        "note_text": "\n".join([n.content for n in record.notes]),
        "labs_text": "\n".join([f"{l.name}: {l.value} {l.unit}" for l in record.labs]),
        "meds_text": ", ".join(record.medications),
        "images": []
    }

elif input_mode == "Paste Text":
    st.header("Mode: Paste Clinical Text")
    st.caption("Paste clinical data below to run the safety audit.")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        note_in = st.text_area("Clinical Note", height=300, placeholder="Example:\nPt presented with weakness...\nPMH: CKD, HTN...")
    with col_p2:
        labs_in = st.text_area("Labs", height=300, placeholder="Example:\nPotassium: 6.0 mmol/L\nCreatinine: 2.1 mg/dL")
    with col_p3:
        meds_in = st.text_area("Medications", height=300, placeholder="Example:\nLisinopril 10mg daily\nSpironolactone 25mg daily")

    standardized_inputs = standardize_input("paste", note_in, labs_in, meds_in)
    standardized_inputs["case_id"] = "USER_PASTE"
    standardized_inputs["images"] = []

elif input_mode == "Upload Files":
    st.header("Mode: Upload Files")
    st.caption("Upload text, PDF, or CSV files. Multiple files are automatically merged.")

    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        note_files = st.file_uploader("Clinical Notes", type=["txt", "pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
        if note_files:
            file_source_type = "MULTIPLE FILES" if len(note_files) > 1 else f"FILE ({note_files[0].name.split('.')[-1].upper()})"
            for f in note_files:
                 if f.name.lower().endswith('.pdf'): file_source_type = "PDF" # Flag as PDF if any PDF present
            st.caption(f"Selected: {len(note_files)} file(s)")
            for f in note_files:
                 st.caption(f"- {f.name}")

    with col_u2:
        labs_files = st.file_uploader("Labs", type=["txt", "csv", "pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
        if labs_files:
            st.caption(f"Selected: {len(labs_files)} file(s)")
            for f in labs_files: st.caption(f"- {f.name}")

    with col_u3:
        meds_files = st.file_uploader("Medications", type=["txt", "pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
        if meds_files:
            st.caption(f"Selected: {len(meds_files)} file(s)")
            for f in meds_files: st.caption(f"- {f.name}")

    standardized_inputs = standardize_input("upload", note_files, labs_files, meds_files)
    standardized_inputs["case_id"] = "USER_UPLOAD"
    standardized_inputs["images"] = []

    # Capture Image Bytes for Multimodal LLM
    all_files = (note_files or []) + (labs_files or []) + (meds_files or [])
    for f in all_files:
        if f.name.lower().endswith((".png", ".jpg", ".jpeg")):
            f.seek(0)
            standardized_inputs["images"].append(f.read())

    # PDF Quality Check
    if file_source_type == "PDF" and standardized_inputs["note_text"]:
        quality_report = analyze_pdf_quality(standardized_inputs["note_text"])

        if not quality_report["quality_pass"]:
            st.warning("⚠️ **Low Quality PDF Text Detected**")
            for w in quality_report["warnings"]:
                st.write(f"- {w}")

            proceed_anyway = st.checkbox("Proceed anyway (Potential risk of missing information)", value=False)
            if not proceed_anyway:
                st.error("Audit blocked due to low-quality extraction. Please provide a cleaner PDF or .txt file.")
                run_btn = False # Disable run


# Tabs
tab_safety, tab_inputs, tab_eval = st.tabs(["🛡️ Safety Review", "📄 Clinical Inputs", "📊 Evaluation"])

# --- Tab 1: Safety Review ---
with tab_safety:
    if run_btn:
        # Validation
        if not standardized_inputs["note_text"] and not standardized_inputs["meds_text"] and not standardized_inputs["labs_text"]:
            st.error("❌ No input data detected. Please provide Note, Labs, or Medications.")
        else:
            # Generate cache key
            def get_prompt_hash():
                try:
                    with open("prompts/extract_facts.md", "r") as f: e_p = f.read()
                    with open("prompts/safety_audit.md", "r") as f: s_p = f.read()
                    return hashlib.md5((e_p + s_p).encode()).hexdigest()[:8]
                except:
                    return "unknown"

            prompt_hash = get_prompt_hash()
            backend_str = st.session_state.backend_type
            model_str = st.session_state.auditor.model_metadata.get("name", "unknown")
            input_hash = hashlib.md5((standardized_inputs["note_text"] + standardized_inputs["labs_text"] + standardized_inputs["meds_text"]).encode()).hexdigest()[:8]

            cache_key = f"{standardized_inputs['case_id']}_{input_hash}_{backend_str}_{model_str}_{prompt_hash}"

            # Initialize cache if not exists
            if "inference_cache" not in st.session_state:
                st.session_state.inference_cache = {}

            # Check cache
            if cache_key in st.session_state.inference_cache:
                cached_data = st.session_state.inference_cache[cache_key]
                st.session_state.last_report = cached_data["report"]
                st.info("⚡ Loaded from cache")
            else:
                with st.spinner("Step 1/2: Extracting clinical facts..."):
                    t0 = time.time()

                    note_text = standardized_inputs["note_text"]
                    labs_text = standardized_inputs["labs_text"]
                    meds_text = standardized_inputs["meds_text"]

                    if st.session_state.get("one_call_mode", False):
                        from src.core.rule_extractor import extract_facts_rule_based
                        extracted_facts = extract_facts_rule_based(note_text, labs_text, meds_text)
                    else:
                        extracted_facts = st.session_state.extractor.extract_facts(
                            note_text, labs_text, meds_text,
                            st.session_state.get("extract_opts")
                        )
                    extract_duration = time.time() - t0

                with st.spinner("Step 2/2: Auditing for safety risks..."):
                    t1 = time.time()
                    report = st.session_state.auditor.run_audit(
                        extracted_facts, note_text, labs_text, meds_text,
                        st.session_state.get("audit_opts"),
                        images=standardized_inputs.get("images")
                    )
                    audit_duration = time.time() - t1
                    total_duration = time.time() - t0

                    if report:
                        if report.metadata is None: report.metadata = {}
                        report.metadata["extract_runtime"] = extract_duration
                        report.metadata["audit_runtime"] = audit_duration
                        report.metadata["total_runtime"] = total_duration

                        # Limit flags
                        max_flags = st.session_state.get("max_flags", 5)
                        if len(report.flags) > max_flags:
                            report.flags = report.flags[:max_flags]
                            report.summary += f" (Showing top {max_flags} flags)"

                        st.session_state.last_report = report
                        st.session_state.last_extracted_facts = extracted_facts

                        st.session_state.inference_cache[cache_key] = {
                            "report": report,
                            "extracted_facts": extracted_facts
                        }
                    else:
                        st.error("Failed to generate safety report")

    if "last_report" in st.session_state:
        report = st.session_state.last_report

        # Summary Card
        flag_count = len(report.flags)
        max_severity = "NONE"
        if flag_count > 0:
            order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            sorted_flags = sorted(report.flags, key=lambda x: order.get(x.severity, 0), reverse=True)
            max_severity = sorted_flags[0].severity

        st.markdown(f"""
        <div class="summary-card">
            <h3>Audit Summary</h3>
            <p><b>{flag_count}</b> Safety Flag(s) Identified</p>
            <p>Highest Severity: <b>{max_severity}</b></p>
            <p style="color: #666; font-size: 0.9em;">{report.summary}</p>
        </div>
        """, unsafe_allow_html=True)

        # Visual / Radiology Report
        if "visual_analysis" in getattr(report, "metadata", {}):
            vis = report.metadata["visual_analysis"]
            st.markdown("### 🖼️ Visual Input Review (Non-Diagnostic)")
            st.info("ℹ️ **Non-diagnostic**: This module checks image quality and completeness only. No clinical interpretation.")

            with st.expander("Visual Observations (No Clinical Interpretation)", expanded=True):
                # 1. Observations
                obs = vis.get("visual_observations", [])
                if obs:
                    st.markdown("**Visual Observations:**")
                    for o in obs: st.markdown(f"- {o}")

                # 2. Quality
                qual = vis.get("quality_issues", [])
                if qual:
                    st.markdown("**Image Quality & Workflow Cues:**")
                    for q in qual: st.markdown(f"- ⚠️ {q}")

                # 3. Risks
                risks = vis.get("workflow_risks", [])
                if risks:
                    st.markdown("**Workflow Risks:**")
                    for r in risks: st.markdown(f"- 🛡️ {r}")

                # 4. Uncertainties
                unc = vis.get("uncertainties", [])
                if unc:
                    st.markdown("**Uncertainties:**")
                    for u in unc: st.markdown(f"- ❓ {u}")

                # 5. Check for "Unavailable" fallback state
                # We identify this by specific text in quality_issues or empty obs
                q_issues = vis.get("quality_issues", [])
                if not obs and any("unavailable" in q.lower() for q in q_issues):
                     st.warning("⚠️ Visual quality review unavailable")

                # Fallback for completely empty clean state
                elif not obs and not qual and not risks and not unc:
                     st.caption("Visual review completed (non-diagnostic). None detected.")

            st.caption("PaliGemma-3b (Google HAI-DEF) • Non-Diagnostic Quality Review Only")

            st.divider()

        elif "visual_error" in getattr(report, "metadata", {}):
            st.warning(f"⚠️ **Visual Analysis Failed**: {report.metadata['visual_error']}")

        # Missing Info
        questions = getattr(report, 'missing_info_questions', [])
        if questions:
            st.markdown("#### ❓ Missing Information / Clarifications")
            for q in questions:
                 st.info(q)

        # Flags Display
        if not report.flags:
            st.success("✅ **No workflow safety issues detected from available structured and visual inputs.**\n\nThis tool does not provide diagnostic interpretation. Verify documentation completeness.")
        else:
            for flag in report.flags:
                # Styles
                sev_color = "red" if flag.severity == "HIGH" else "orange" if flag.severity == "MEDIUM" else "green"
                css_class = f"severity-{flag.severity.value.lower()}" if hasattr(flag.severity, 'value') else f"severity-{str(flag.severity).lower()}"

                cat_val = flag.category.value if hasattr(flag.category, 'value') else str(flag.category)
                display_cat = cat_val.replace("_", " ").title()

                sev_val = flag.severity.value if hasattr(flag.severity, 'value') else str(flag.severity)
                if "SafetySeverity." in sev_val: sev_val = sev_val.split(".")[-1]

                with st.container():
                    st.markdown(f"""
                    <div class="{css_class}">
                        <h4><span style='color:{sev_color}'>[{sev_val}]</span> {display_cat}</h4>
                        <p><b>{flag.explanation}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("Show Evidence", expanded=True):
                        st.caption("Verbatim quotes from record:")
                        for ev in flag.evidence:
                             quote = ev.quote if hasattr(ev, 'quote') else ev.get('quote')
                             src = getattr(ev, 'source', 'UNKNOWN') if hasattr(ev, 'source') else ev.get('source', 'UNKNOWN')

                             badge_color = "#e0e0e0"
                             if src == "NOTE": badge_color = "#e3f2fd"
                             elif src == "LABS": badge_color = "#fbe9e7"
                             elif src == "MEDS": badge_color = "#e8f5e9"

                             st.markdown(f"""
                             <div style="background-color: #f8f9fa; border-left: 3px solid #dfe2e5; padding: 8px; margin: 5px 0; font-family: monospace; font-size: 0.9em; color: #444;">
                                <span style="background-color: {badge_color}; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.8em; margin-right: 5px;">{src}</span>
                                "{quote}"
                             </div>
                             """, unsafe_allow_html=True)

                    with st.expander("🔍 Click for Detailed Logic Breakdown"):
                        if flag.reasoning:
                            st.markdown(flag.reasoning.replace("SITUATION:", "**SITUATION:**").replace("MECHANISM:", "\n\n**MECHANISM:**").replace("ASSESSMENT:", "\n\n**ASSESSMENT:**"))
                        else:
                            st.write(flag.explanation)

                        if flag.recommendation:
                            st.info(f"💡 **Review Guidance**: {flag.recommendation}")

    else:
        st.info("Click 'Run Safety Review' in the sidebar to analyze this case.")

# --- Tab 2: Inputs ---
with tab_inputs:
    col1, col2 = st.columns(2)

    # Highlights Setup
    highlights = []
    if "last_report" in st.session_state:
        for flag in st.session_state.last_report.flags:
            for ev in flag.evidence:
                quote = getattr(ev, 'quote', None) or (ev.get('quote') if isinstance(ev, dict) else str(ev))
                if quote: highlights.append(quote)

    with col1:
        st.subheader("Clinical Note")

        # Source Metadata
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
        <div style="height: 500px; overflow-y: auto; background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-family: monospace; white-space: pre-wrap;">
        {display_content if display_content else "No clinical note provided."}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Structured Data")

        # If we have a structured record (Demo Mode), show dataframes.
        # Otherwise show the parsed text for Labs/Meds.

        if record:
            # --- Demo Mode (Structured) ---
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
            # --- User Input Mode (Text-based) ---
            st.info("Displaying parsed input text used for analysis (Structured view not available for raw input).")

            st.markdown("**Medications (Text)**")
            st.text_area("Meds", standardized_inputs["meds_text"], height=150, disabled=True)

            st.markdown("**Laboratories (Text)**")
            st.text_area("Labs", standardized_inputs["labs_text"], height=200, disabled=True)

# --- Tab 3: Evaluation ---
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
