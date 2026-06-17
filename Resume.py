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
  .info-badge { background-color: rgba(56, 189, 248, 0.2); border: 1px solid #38bdf8; color: #38bdf8; padding: 2px 8px; border-radius: 10px; font-size: 0.8rem; margin-right: 10px;}
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
    """Uses Gemini to extract structured JSON data highlighting match, skills gap, and red flags."""
    genai.configure(api_key=api_key)
    # Using gemini-2.5-flash for fast, structured extraction
    model = genai.GenerativeModel("gemini-3.1-flash")
    current_date = date.today().strftime("%B %Y")
    
    prompt = f"""
    You are an enterprise-grade AI ATS (Applicant Tracking System) parser and evaluator.
    Analyze the Candidate's Resume against the provided Job Description (JD).
    
    Current Date: {current_date} (Use this to accurately calculate 'Present' in job durations).
    
    Job Description:
    {jd_text}
    
    Resume Text:
    {resume_text}
    
    Instructions:
    1. candidate_name: Extract the candidate's full name.
    2. current_role: Extract their current or most recent job title.
    3. location: Extract their current city/location (if available).
    4. total_experience_years: Calculate exact total professional experience as a float (e.g., 3.6). Do NOT overcount overlapping roles.
    5. education: Extract academic degrees, institutions, and scores.
    6. Skills Analysis: 
       - matched_jd_skills: Skills they have that are explicitly requested in the JD.
       - missing_jd_skills: Core skills requested in the JD that are nowhere to be found in the resume.
    7. red_flags: List obvious dealbreakers (e.g., "JD requires 5 years exp, candidate has 2", "Missing mandatory Bachelor's degree", "Based in NY, JD requires London").
    8. overall_match_percentage: Give a strict integer (0-100) reflecting how well the candidate fits the JD. Be critical.

    Respond STRICTLY with a JSON object matching this schema:
    {{
        "candidate_name": "string",
        "current_role": "string",
        "location": "string",
        "total_experience_years": 0.0,
        "education": [
            {{
                "degree": "string",
                "institution": "string",
                "score": "string"
            }}
        ],
        "matched_jd_skills": ["string"],
        "missing_jd_skills": ["string"],
        "red_flags": ["string"],
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
    """Renders the main ATS candidate profiles."""
    st.markdown("### 📋 Candidate Evaluation Profiles")
    
    # Sort results by match percentage descending
    results.sort(key=lambda x: x['overall_match_percentage'], reverse=True)
    
    for res in results:
        # Expander Title with Match Score
        expander_title = f"{res['candidate_name']} - {res['overall_match_percentage']}% Match ({res['total_experience_years']} Yrs Exp)"
        
        with st.expander(expander_title):
            # Display Quick Logistics
            location = res.get('location', 'Location N/A')
            role = res.get('current_role', 'Role N/A')
            st.markdown(f"<span class='info-badge'>📍 {location}</span> <span class='info-badge'>💼 {role}</span><br><br>", unsafe_allow_html=True)
            
            # Show Red Flags prominently if they exist
            if res.get('red_flags'):
                st.error("**🚩 Potential Red Flags & Missing Criteria:**\n" + "\n".join([f"- {flag}" for flag in res['red_flags']]))
            
            sc1, sc2 = st.columns([1, 1])
            
            with sc1:
                st.markdown("**🎓 Education Details**")
                if not res.get('education'):
                    st.write("No formal education parsed.")
                else:
                    for edu in res['education']:
                        st.markdown(f"- **{edu.get('degree', 'Unknown Degree')}** <br> <span style='color:#94a3b8; font-size:0.9em;'>{edu.get('institution', 'Unknown Inst.')} | {edu.get('score', '')}</span>", unsafe_allow_html=True)
            
            with sc2:
                st.markdown("**🎯 Skills Gap Analysis**")
                
                st.markdown("*Matched JD Requirements:*")
                if res.get('matched_jd_skills'):
                    matched_html = "".join([f"<span class='skill-pill-match'>{s}</span>" for s in res['matched_jd_skills']])
                    st.markdown(matched_html, unsafe_allow_html=True)
                else:
                    st.write("None detected.")
                
                st.markdown("<br>*Missing JD Requirements:*", unsafe_allow_html=True)
                if res.get('missing_jd_skills'):
                    missed_html = "".join([f"<span class='skill-pill-miss'>{s}</span>" for s in res['missing_jd_skills']])
                    st.markdown(missed_html, unsafe_allow_html=True)
                else:
                    st.write("None detected.")

# ─── Main App Flow ────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown("### ⚙️ ATS Engine Settings")
        api_key = st.text_input("Gemini API Key", type="password", help="Required to run the parsing LLM.")
        if not api_key and "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
            
        st.markdown("---")
        st.markdown("### 📄 Input Documents")
        jd_file = st.file_uploader("Upload Job Description (JD)", type=["pdf", "docx", "txt"])
        resume_files = st.file_uploader("Upload Resumes (Batch)", type=["pdf", "docx"], accept_multiple_files=True)
        
        process_btn = st.button("🚀 Run AI Analysis", use_container_width=True, type="primary")

    if not api_key:
        st.info("👋 Welcome to the AI Matcher! Please enter your Google Gemini API Key in the sidebar to securely process documents.")
        return

    if process_btn:
        if not jd_file:
            st.warning("Please upload a Job Description to evaluate candidates against.")
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
            status_text.text(f"Extracting & Evaluating {resume_file.name} ({i+1}/{len(resume_files)})...")
            resume_text = extract_text_from_file(resume_file)
            
            # Analyze via Gemini using the enhanced prompt
            parsed_data = analyze_resume_with_gemini(resume_text, jd_text, api_key)
            
            if parsed_data:
                # If name is missing or defaulted by LLM, fallback to filename
                if not parsed_data.get("candidate_name") or parsed_data["candidate_name"].lower() == "string":
                    parsed_data["candidate_name"] = resume_file.name.replace(".pdf", "").replace(".docx", "")
                all_results.append(parsed_data)
                
            progress_bar.progress((i + 1) / len(resume_files))
            
        status_text.empty()
        progress_bar.empty()
        
        if all_results:
            render_dashboard(all_results)
        else:
            st.error("Extraction failed for all resumes. Please check your API key validity and document contents.")

if __name__ == "__main__":
    main()
