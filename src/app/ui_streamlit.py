import streamlit as st
import json
import os
from src.core.audit import SafetyAuditor
from src.core.schema import PatientRecord

st.set_page_config(page_title="SentinelMD", layout="wide")
st.warning("⚠️ **ADVISORY ONLY**: Not for clinical diagnosis/treatment.")
st.title("SentinelMD: Clinical Safety Copilot")

# Load Synthetic Cases
DATA_DIR = "data/synthetic"
cases = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
cases.sort()

with st.sidebar:
    st.header("Case Selection")
    selected_case = st.selectbox("Select Patient Case", cases)

    if st.button("Load Case"):
        with open(os.path.join(DATA_DIR, selected_case), "r") as f:
            data = json.load(f)
            # Validate against schema
            st.session_state.record = PatientRecord(**data)
            st.success(f"Loaded {selected_case}")

if "record" in st.session_state:
    record: PatientRecord = st.session_state.record

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Patient Card")
        st.metric("ID", record.patient_id)
        st.write(f"**Allergies:** {record.allergies}")
        st.write(f"**Meds:** {record.medications}")
        st.json(record.vitals)

    with col2:
        st.subheader("Clinical Notes")
        for note in record.notes:
            st.write(f"**{note.date}**: {note.content}")

    st.markdown("---")
    if st.button("🛡️ Run Safety Audit", type="primary"):
        auditor = SafetyAuditor()
        report = auditor.audit(record)

        st.subheader("Audit Report")
        if not report.observations:
            st.success("No safety risks flagged.")
        else:
            for obs in report.observations:
                color = "red" if obs.severity in ["HIGH", "CRITICAL"] else "orange"
                st.markdown(f":{color}[**{obs.severity}**] - {obs.explanation}")
                st.caption(f"Evidence: \"{obs.evidence}\"")
                st.info(f"Advisory: {obs.recommendation}")
else:
    st.info("Select a case to begin.")
