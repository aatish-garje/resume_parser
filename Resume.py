"""
Universal AI Resume Matcher & ATS Screening Platform
Multi-Model Edition — Powered by Gemini, Groq, and ChatGPT
"""

import streamlit as st
import pdfplumber
import docx
import json
import logging
from datetime import date
import io

# AI Provider Imports
import google.generativeai as genai
from groq import Groq
from openai import OpenAI

# ─── Configuration & Styling ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal AI ATS",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# ─── Model Mappings ───────────────────────────────────────────────────────────
GEMINI_MODELS = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
    "Gemini 2 Flash": "gemini-2.0-flash",
    "Gemini 2 Flash Lite": "gemini-2.0-flash-lite",
    "Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite",
    "Gemini 3.5 Flash": "gemini-3.5-flash"
}

GROQ_MODELS = {
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
    "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
    "Llama 3 70B": "llama3-70b-8192",
    "Llama 3 8B": "llama3-8b-8192",
    "Gemma 2 9B IT": "gemma2-9b-it",
    "Mixtral 8x7B": "mixtral-8x7b-32768"
}

OPENAI_MODELS = {
    "GPT-4o": "gpt-4o",
    "GPT-4o Mini": "gpt-4o-mini",
    "GPT-4 Turbo": "gpt-4-turbo",
    "GPT-3.5 Turbo": "gpt-3.5-turbo"
}

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
            text = uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        logging.error(f"Error reading {uploaded_file.name}: {e}")
    return text.strip()

# ─── Shared Prompt Logic ──────────────────────────────────────────────────────
def generate_evaluation_prompt(resume_text: str, jd_text: str) -> str:
    current_date = date.today().strftime("%B %Y")
    return f"""
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

    Respond STRICTLY with a JSON object matching this schema. Do not include markdown formatting like ```json in the output, just raw JSON:
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

# ─── AI Extraction Logic ──────────────────────────────────────────────────────
def analyze_with_gemini(resume_text: str, jd_text: str, api_key: str, model_name: str) -> dict:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = generate_evaluation_prompt(resume_text, jd_text)
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1}
        )
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:-3].strip()
            
        return json.loads(response_text)
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return None

def analyze_with_groq(resume_text: str, jd_text: str, api_key: str, model_name: str) -> dict:
    client = Groq(api_key=api_key)
    prompt = generate_evaluation_prompt(resume_text, jd_text)
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful HR assistant that always outputs valid JSON."},
                {"role": "user", "content": prompt}
            ],
            model=model_name,
            temperature=0.1,
            response_format={"type": "json_object"} 
        )
        
        response_text = response.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        st.error(f"Groq API Error: {e}")
        return None

def analyze_with_openai(resume_text: str, jd_text: str, api_key: str, model_name: str) -> dict:
    client = OpenAI(api_key=api_key)
    prompt = generate_evaluation_prompt(resume_text, jd_text)
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful HR assistant that always outputs valid JSON."},
                {"role": "user", "content": prompt}
            ],
            model=model_name,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        return None

# ─── UI Rendering ─────────────────────────────────────────────────────────────
def render_dashboard(results: list):
    """Renders the main ATS candidate profiles."""
    st.markdown("### 📋 Candidate Evaluation Profiles")
    
    # Sort results by match percentage descending
    results.sort(key=lambda x: x.get('overall_match_percentage', 0), reverse=True)
    
    for res in results:
        match_score = res.get('overall_match_percentage', 0)
        name = res.get('candidate_name', 'Unknown')
        exp = res.get('total_experience_years', 0)
        
        expander_title = f"{name} - {match_score}% Match ({exp} Yrs Exp)"
        
        with st.expander(expander_title):
            location = res.get('location', 'Location N/A')
            role = res.get('current_role', 'Role N/A')
            st.markdown(f"<span class='info-badge'>📍 {location}</span> <span class='info-badge'>💼 {role}</span><br><br>", unsafe_allow_html=True)
            
            if res.get('red_flags'):
                st.error("**🚩 Potential Red Flags & Missing Criteria:**\n" + "\n".join([f"- {flag}" for flag in res['red_flags']]))
            
            sc1, sc2 = st.columns([1, 1])
            
            with sc1:
                st.markdown("**🎓 Education Details**")
                if not res.get('education'):
                    st.write("No formal education parsed.")
                else:
                    for edu in res['education']:
                        st.markdown(f"- **{edu.get('degree', 'Unknown')}** <br> <span style='color:#94a3b8; font-size:0.9em;'>{edu.get('institution', 'Unknown')} | {edu.get('score', '')}</span>", unsafe_allow_html=True)
            
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
        
        provider = st.selectbox(
            "Select AI Provider", 
            ["Gemini", "Groq", "ChatGPT (OpenAI)"]
        )
        
        if provider == "Gemini":
            selected_model_name = st.selectbox("Select Model", list(GEMINI_MODELS.keys()), index=0)
            actual_model = GEMINI_MODELS[selected_model_name]
            api_key = st.text_input("Gemini API Key", type="password", help="Required to run the parsing LLM.")
        elif provider == "Groq":
            selected_model_name = st.selectbox("Select Model", list(GROQ_MODELS.keys()), index=0)
            actual_model = GROQ_MODELS[selected_model_name]
            api_key = st.text_input("Groq API Key (Free)", type="password", help="Get a free key at console.groq.com")
        else:
            selected_model_name = st.selectbox("Select Model", list(OPENAI_MODELS.keys()), index=0)
            actual_model = OPENAI_MODELS[selected_model_name]
            api_key = st.text_input("OpenAI API Key", type="password")

        st.markdown("---")
        st.markdown("### 📄 Input Documents")
        jd_file = st.file_uploader("Upload Job Description (JD)", type=["pdf", "docx", "txt"])
        resume_files = st.file_uploader("Upload Resumes (Batch)", type=["pdf", "docx"], accept_multiple_files=True)
        
        process_btn = st.button("🚀 Run AI Analysis", use_container_width=True, type="primary")

    if not api_key:
        st.info(f"👋 Welcome to the AI Matcher! Please enter your {provider} API Key in the sidebar to securely process documents.")
        return

    if process_btn:
        if not jd_file:
            st.warning("Please upload a Job Description to evaluate candidates against.")
            return
        if not resume_files:
            st.warning("Please upload at least one Resume.")
            return
            
        jd_text = extract_text_from_file(jd_file)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        all_results = []
        
        for i, resume_file in enumerate(resume_files):
            status_text.text(f"Extracting & Evaluating {resume_file.name} ({i+1}/{len(resume_files)})...")
            resume_text = extract_text_from_file(resume_file)
            
            parsed_data = None
            if provider == "Gemini":
                parsed_data = analyze_with_gemini(resume_text, jd_text, api_key, actual_model)
            elif provider == "Groq":
                parsed_data = analyze_with_groq(resume_text, jd_text, api_key, actual_model)
            else:
                parsed_data = analyze_with_openai(resume_text, jd_text, api_key, actual_model)
            
            if parsed_data:
                if not parsed_data.get("candidate_name") or parsed_data["candidate_name"].lower() in ["string", "unknown"]:
                    parsed_data["candidate_name"] = resume_file.name.rsplit(".", 1)[0]
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
