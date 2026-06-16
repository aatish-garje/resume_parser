"""
AI Resume Matcher & ATS Screening Platform
Production-Grade Edition — Powered by Gemini 1.5 Flash Structured Extraction
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pdfplumber
import docx
import json
import logging
from datetime import date
import google.generativeai as genai
import io

# ─── Configuration & Styling ──────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Matcher & ATS",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAPER_BG = "rgba(0,0,0,0)"
CHART_BG = "rgba(0,0,0,0)"
FONT_COLOR = "#e2e8f0"

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%); color: #f8fafc; }
  .kpi-card { 
      background: rgba(255, 255, 255, 0.05); 
      border: 1px solid rgba(255, 255, 255, 0.1); 
      border-radius: 10px; 
      padding: 20px; 
      margin-bottom: 20px;
  }
  .skill-pill-match { background-color: rgba(52, 211, 153, 0.2); border: 1px solid #34d399; color: #34d399; padding: 4px 10px; border-radius: 15px; font-size: 0.85rem; display: inline-block; margin: 2px; }
  .skill-pill-miss { background-color: rgba(248, 113, 113, 0.2); border: 1px solid #f87171; color: #f87171; padding: 4px 10px; border-radius: 15px; font-size: 0.85rem; display: inline-block; margin: 2px; }
  div[data-testid="stExpander"] { background: rgba(255,255,255,0.02); border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# ─── File Extraction Utilities ────────────────────────────────────────────────
def extract_text_from_file(uploaded_file) -> str:
    """Extracts text from PDF or DOCX files."""
    text = ""
    try:
        if uploaded_file.name.lower().endswith('.pdf'):
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        elif uploaded_file.name.lower().endswith('.docx'):
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            # Fallback for txt
            text = uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        logging.error(f"Error reading {uploaded_file.name}: {e}")
    return text.strip()

# ─── Gemini LLM Extraction Logic ──────────────────────────────────────────────
def analyze_resume_with_gemini(resume_text: str, jd_text: str, api_key: str) -> dict:
    """Uses Gemini 1.5 Flash to extract structured JSON data from the resume based on the JD."""
    genai.configure(api_key=api_key)
    # Changed to '-latest' to resolve 404 errors on some API endpoints/older SDK versions
    model = genai.GenerativeModel("gemini-2.5-flash")
    current_date = date.today().strftime("%B %Y")
    
    prompt = f"""
    You are an enterprise-grade AI ATS (Applicant Tracking System).
    Analyze the Candidate's Resume against the provided Job Description (JD).
    
    Current Date: {current_date} (Use this to accurately calculate 'Present' in job durations).
    
    Job Description:
    {jd_text}
    
    Resume Text:
    {resume_text}
    
    Instructions:
    1. candidate_name: Extract the candidate's full name.
    2. total_experience_years: Calculate the exact total professional experience as a float (e.g., 3.6). Add up all valid job durations up to the current date. Do NOT overcount overlapping roles.
    3. education: Extract genuine academic degrees, institutions, and scores. Ignore generic links like LinkedIn or GitHub.
    4. Skills: 
       - Extract ONLY genuine technical, domain, or professional skills. 
       - Do NOT include geographic locations (e.g., 'Mumbai'), random verbs, or generic nouns (e.g., 'Requirements', 'Responsibilities').
       - Cross-reference with the JD to categorize into 'matched_jd_skills' and 'missing_jd_skills'.
    5. overall_match_percentage: Give a strict integer (0-100) reflecting how well the candidate fits the JD based on skills and experience.

    Respond STRICTLY with a JSON object matching this schema:
    {{
        "candidate_name": "string",
        "total_experience_years": 0.0,
        "education": [
            {{
                "degree": "string",
                "institution": "string",
                "score": "string"
            }}
        ],
        "extracted_skills": ["string"],
        "matched_jd_skills": ["string"],
        "missing_jd_skills": ["string"],
        "overall_match_percentage": 0
    }}
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1}
        )
        
        # Strip potential markdown formatting that can sometimes wrap JSON responses
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:-3].strip()
            
        return json.loads(response_text)
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return None

# ─── UI Rendering ─────────────────────────────────────────────────────────────
def render_dashboard(results: list):
    """Renders the main ATS dashboard."""
    st.markdown("### 📊 ATS Analytics Dashboard")
    
    # Summary Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="kpi-card">
            <h4 style="margin:0; color:#94a3b8;">Total Candidates</h4>
            <h1 style="margin:0; color:#38bdf8;">{len(results)}</h1>
            </div>""", unsafe_allow_html=True)
    with col2:
        avg_match = sum(r['overall_match_percentage'] for r in results) / len(results) if results else 0
        st.markdown(f"""<div class="kpi-card">
            <h4 style="margin:0; color:#94a3b8;">Average Match</h4>
            <h1 style="margin:0; color:#34d399;">{avg_match:.1f}%</h1>
            </div>""", unsafe_allow_html=True)
    with col3:
        top_candidates = sum(1 for r in results if r['overall_match_percentage'] >= 75)
        st.markdown(f"""<div class="kpi-card">
            <h4 style="margin:0; color:#94a3b8;">Highly Qualified (>75%)</h4>
            <h1 style="margin:0; color:#fbbf24;">{top_candidates}</h1>
            </div>""", unsafe_allow_html=True)

    # Convert results to DataFrame for charting
    df = pd.DataFrame(results)
    
    # Layout for charts
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # Match Percentage Bar Chart
        df_sorted = df.sort_values(by='overall_match_percentage', ascending=True)
        fig_match = px.bar(
            df_sorted, 
            x='overall_match_percentage', 
            y='candidate_name', 
            orientation='h',
            color='overall_match_percentage',
            color_continuous_scale=[[0, "#f87171"], [0.5, "#fbbf24"], [1, "#34d399"]],
            title="Candidate Match Scores"
        )
        fig_match.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=CHART_BG, font_color=FONT_COLOR, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_match, use_container_width=True)

    with c2:
        # Experience Distribution
        fig_exp = px.box(
            df, 
            y='total_experience_years', 
            points="all", 
            title="Experience Distribution (Years)",
            color_discrete_sequence=["#38bdf8"]
        )
        fig_exp.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=CHART_BG, font_color=FONT_COLOR, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_exp, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Detailed Candidate Profiles")
    
    # Sort results by match percentage descending
    results.sort(key=lambda x: x['overall_match_percentage'], reverse=True)
    
    for res in results:
        with st.expander(f"{res['candidate_name']} - {res['overall_match_percentage']}% Match ({res['total_experience_years']} Years Exp)"):
            sc1, sc2 = st.columns([1, 1])
            
            with sc1:
                st.markdown("**🎓 Education**")
                if not res['education']:
                    st.write("No formal education parsed.")
                for edu in res['education']:
                    st.markdown(f"- **{edu.get('degree', 'Unknown Degree')}** <br> <span style='color:#94a3b8; font-size:0.9em;'>{edu.get('institution', 'Unknown Inst.')} | {edu.get('score', '')}</span>", unsafe_allow_html=True)
            
            with sc2:
                st.markdown("**🎯 Matched Requirements**")
                matched_html = "".join([f"<span class='skill-pill-match'>{s}</span>" for s in res['matched_jd_skills']])
                st.markdown(matched_html, unsafe_allow_html=True)
                
                st.markdown("<br>**⚠️ Missing Requirements**", unsafe_allow_html=True)
                missed_html = "".join([f"<span class='skill-pill-miss'>{s}</span>" for s in res['missing_jd_skills']])
                st.markdown(missed_html, unsafe_allow_html=True)

# ─── Main App Flow ────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown("### ⚙️ ATS Settings")
        api_key = st.text_input("Gemini API Key", type="password", help="Get this from Google AI Studio")
        if not api_key and "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
            
        st.markdown("---")
        st.markdown("### 📄 Documents")
        jd_file = st.file_uploader("Upload Job Description", type=["pdf", "docx", "txt"])
        resume_files = st.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)
        
        process_btn = st.button("🚀 Run AI Analysis", use_container_width=True, type="primary")

    if not api_key:
        st.info("👋 Welcome! Please enter your Google Gemini API Key in the sidebar to start parsing resumes intelligently.")
        return

    if process_btn:
        if not jd_file:
            st.warning("Please upload a Job Description.")
            return
        if not resume_files:
            st.warning("Please upload at least one Resume.")
            return
            
        jd_text = extract_text_from_file(jd_file)
        
        # UI for processing state
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        
        for i, resume_file in enumerate(resume_files):
            status_text.text(f"Analyzing {resume_file.name} ({i+1}/{len(resume_files)})...")
            resume_text = extract_text_from_file(resume_file)
            
            # Analyze via Gemini
            parsed_data = analyze_resume_with_gemini(resume_text, jd_text, api_key)
            
            if parsed_data:
                # If name is missing, use filename
                if not parsed_data.get("candidate_name") or parsed_data["candidate_name"].lower() == "string":
                    parsed_data["candidate_name"] = resume_file.name.replace(".pdf", "").replace(".docx", "")
                all_results.append(parsed_data)
                
            progress_bar.progress((i + 1) / len(resume_files))
            
        status_text.empty()
        progress_bar.empty()
        
        if all_results:
            render_dashboard(all_results)
        else:
            st.error("Extraction failed for all resumes. Please check API key and file contents.")

if __name__ == "__main__":
    main()
