import asyncio
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback
from agents.orchestrator import ResumeOrchestrator
from config.settings import settings
import shutil

# --- Config ---
st.set_page_config(page_title="AI Resume Matcher (Gemini 2.5)", layout="wide")
st.title("🎯 AI Resume Matcher")
st.markdown("### Powered by Strands Agents & Gemini 2.5")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    api_key = st.text_input("Google API Key", value=settings.GOOGLE_API_KEY, type="password")
    st.badge("Key provided for Demo, will be set to empty value", icon=":material/settings:", color="orange")
    if api_key:
        settings.GOOGLE_API_KEY = api_key
        os.environ["GOOGLE_API_KEY"] = api_key # Update env for subprocesses if any

    st.markdown("---")
    st.subheader("📊 Retrieval Parameters")
    top_n = st.number_input("Candidates to Retrieve", 1, 50, 5)
    settings.TOP_N_CANDIDATES = top_n

    st.markdown("---")
    st.markdown("**Model Info**")
    st.info(f"Extractor: {settings.EXTRACTION_MODEL}")
    st.info(f"Matcher: {settings.RERANK_MODEL}")
    st.info(f"Embedder: {settings.EMBEDDING_MODEL_ID}")

# --- Layout ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("1️⃣ Job Description")
    if st.radio("Input", ["Text", "Sample"]) == "Text":
        job_description = st.text_area("Paste JD...", height=300)
    else:
        try:
            with open("evaluation/dataset/job_description.txt") as f:
                job_description = f.read()
            st.text_area("Sample JD", job_description, height=300, disabled=True)
        except FileNotFoundError:
            st.error("Sample JD not found.")
            job_description = ""

with c2:
    st.subheader("2️⃣ Resumes")
    resume_paths = []

    if st.radio("Source", ["Upload", "Synthetic"]) == "Upload":
        uploaded_files = st.file_uploader("Upload PDF/TXT/DOCX", accept_multiple_files=True)
        if uploaded_files:
            os.makedirs("temp_resumes", exist_ok=True)
            for uf in uploaded_files:
                path = f"temp_resumes/{uf.name}"
                with open(path, "wb") as f: f.write(uf.getbuffer())
                resume_paths.append(path)
            st.success(f"Loaded {len(uploaded_files)} files.")
    else:
        # Load synthetic resumes
        folder = "evaluation/dataset/resumes"
        if os.path.exists(folder):
            resume_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.pdf', '.docx', '.txt'))]
            st.info(f"Using {len(resume_paths)} synthetic resumes.")
        else:
            st.error("Synthetic dataset not found.")

# --- Helper Functions ---
def format_score(score):
    return f"{score * 100:.1f}%"

def get_score_color(score):
    if score >= 0.8: return "🟢"
    elif score >= 0.6: return "🟡"
    else: return "🔴"

# --- Pipeline ---
async def main_pipeline():
    try:
        # Check API Key
        if not settings.GOOGLE_API_KEY:
            st.error("❌ Google API Key is required. Please enter it in the sidebar.")
            return

        orch = ResumeOrchestrator()

        with st.spinner("🔄 Processing Pipeline..."):
            # Run
            # If resume_paths is empty, it will just search existing index
            results = await orch.process_resumes(job_description, resume_paths)

        if not results:
            st.warning("⚠️ No matches found.")
        else:
            if len(results) > settings.TOP_N_CANDIDATES:
                results = results[:settings.TOP_N_CANDIDATES]
                
            st.success(f"✅ Found {len(results)} candidates.")
            
            # Stats
            stats = orch.get_summary_statistics(results)
            st.metric("Top Candidate", stats['top_candidate'], format_score(stats['top_candidate_score']))

            st.markdown("---")
            st.subheader("🏆 Ranked Candidates")

            for idx, res in enumerate(results, 1):
                final_score = res.get('final_score', 0)
                name = res.get('candidate_name', 'Unknown')
                
                with st.expander(f"**#{idx} {get_score_color(final_score)} {name}** - {format_score(final_score)}"):
                    
                    st.markdown("#### 📊 Score Breakdown")
                    st.code(res.get('breakdown', 'N/A'))
                    
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.markdown("**✅ Strengths**")
                        for s in res.get('strengths', []):
                            st.markdown(f"- {s}")
                    with c_b:
                        st.markdown("**⚠️ Gaps**")
                        for g in res.get('gaps', []):
                            st.markdown(f"- {g}")
                    
                    st.markdown("**💭 Analysis**")
                    st.write(res.get('reasoning', ''))

            # CSV Download
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.code(traceback.format_exc())

# --- Main Execution ---
if st.button("🚀 Start Matching", type="primary", use_container_width=True):
    if not job_description:
        st.error("❌ Job Description needed.")
    else:
        asyncio.run(main_pipeline())
