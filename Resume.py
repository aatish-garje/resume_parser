"""
AI Resume Matcher & ATS Screening Platform
Production-grade single-file Streamlit application — v2
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pdfplumber
import docx
import re
import json
import io
import hashlib
import logging
from datetime import datetime, date
from collections import Counter
from typing import Optional
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# ─── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Matcher & ATS Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  html,body,[class*="css"]{font-family:'Inter',sans-serif;}
  .stApp{background:linear-gradient(135deg,#0f1117 0%,#1a1d2e 100%);}
  .kpi-card{background:linear-gradient(135deg,#1e2235 0%,#252a40 100%);border:1px solid rgba(99,102,241,0.25);border-radius:16px;padding:20px 24px;text-align:center;box-shadow:0 4px 24px rgba(0,0,0,0.4);transition:transform 0.2s,box-shadow 0.2s;}
  .kpi-card:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(99,102,241,0.2);}
  .kpi-value{font-size:2.2rem;font-weight:700;color:#818cf8;line-height:1;}
  .kpi-label{font-size:0.8rem;color:#94a3b8;margin-top:6px;text-transform:uppercase;letter-spacing:0.08em;}
  .kpi-delta{font-size:0.75rem;color:#34d399;margin-top:4px;}
  .section-header{font-size:1.4rem;font-weight:700;color:#e2e8f0;border-left:4px solid #6366f1;padding-left:14px;margin:28px 0 16px 0;}
  .skill-tag{display:inline-block;padding:3px 10px;border-radius:6px;font-size:0.75rem;font-weight:500;margin:2px;}
  .tag-match{background:rgba(52,211,153,0.15);color:#34d399;border:1px solid rgba(52,211,153,0.3);}
  .tag-missing{background:rgba(248,113,113,0.15);color:#f87171;border:1px solid rgba(248,113,113,0.3);}
  .tag-extra{background:rgba(99,102,241,0.15);color:#818cf8;border:1px solid rgba(99,102,241,0.3);}
  .tag-critical{background:rgba(239,68,68,0.2);color:#fca5a5;border:1px solid rgba(239,68,68,0.5);font-weight:700;}
  .badge-excellent{background:linear-gradient(135deg,#059669,#10b981);color:#fff;padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;}
  .badge-strong{background:linear-gradient(135deg,#0284c7,#38bdf8);color:#fff;padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;}
  .badge-good{background:linear-gradient(135deg,#7c3aed,#a78bfa);color:#fff;padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;}
  .badge-average{background:linear-gradient(135deg,#b45309,#fbbf24);color:#fff;padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;}
  .badge-poor{background:linear-gradient(135deg,#991b1b,#f87171);color:#fff;padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;}
  .ai-panel{background:linear-gradient(135deg,rgba(99,102,241,0.08) 0%,rgba(139,92,246,0.08) 100%);border:1px solid rgba(99,102,241,0.3);border-radius:14px;padding:22px;}
  .ai-section-title{font-size:0.85rem;font-weight:600;color:#818cf8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;}
  .stProgress>div>div>div{background:linear-gradient(90deg,#6366f1,#8b5cf6)!important;}
  [data-testid="metric-container"]{background:#1e2235;border:1px solid rgba(99,102,241,0.2);border-radius:10px;padding:14px;}
  .stTabs [data-baseweb="tab-list"]{gap:8px;background:transparent;}
  .stTabs [data-baseweb="tab"]{background:#1e2235;border-radius:8px;color:#94a3b8;border:1px solid rgba(99,102,241,0.15);padding:8px 20px;font-weight:500;}
  .stTabs [aria-selected="true"]{background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;color:white!important;border-color:transparent!important;}
  [data-testid="stFileUploader"]{background:rgba(30,34,53,0.6);border:2px dashed rgba(99,102,241,0.35);border-radius:12px;padding:8px;}
  [data-testid="stSidebar"]{background:#13162a;border-right:1px solid rgba(99,102,241,0.15);}
  .hero-title{font-size:2.4rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#c084fc,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.15;margin-bottom:6px;}
  .hero-sub{font-size:1rem;color:#64748b;margin-bottom:24px;}
  .divider{height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,0.4),transparent);margin:20px 0;}
  .compare-header{background:#1e2235;border-radius:10px;padding:10px;text-align:center;font-weight:600;color:#818cf8;}
  .ats-suggestion{background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.25);border-radius:8px;padding:10px 14px;margin:4px 0;font-size:0.85rem;color:#fde68a;}
  .resume-suggestion{background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);border-radius:8px;padding:10px 14px;margin:4px 0;font-size:0.85rem;color:#c7d2fe;}
  .critical-missing{background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.35);border-radius:8px;padding:10px 14px;margin:4px 0;font-size:0.85rem;color:#fca5a5;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────────
CHART_BG   = "rgba(0,0,0,0)"
PAPER_BG   = "rgba(0,0,0,0)"
FONT_COLOR = "#e2e8f0"
GRID_COLOR = "rgba(99,102,241,0.15)"
CHART_COLORS = ["#6366f1","#8b5cf6","#06b6d4","#34d399","#fbbf24","#f87171","#e879f9","#fb923c","#4ade80","#60a5fa"]

WEIGHTS = {
    "skill_match": 0.40,
    "semantic_match": 0.25,
    "experience_match": 0.15,
    "ats_score": 0.10,
    "education_match": 0.10,
}

DEGREE_KEYWORDS = [
    "phd","ph.d","doctor","m.tech","mtech","m.e.","m.sc","msc",
    "m.s.","ms ","mba","b.tech","btech","b.e.","be ","b.sc","bsc",
    "bachelor","master","degree","diploma","associate","b.com","bcom",
]
CERT_KEYWORDS = [
    "certified","certification","certificate","aws","azure","gcp",
    "pmp","cissp","ceh","comptia","oracle","cisco","microsoft","google cloud",
    "kubernetes","docker certified","scrum","agile","itil","six sigma",
]

TECH_PATTERNS = [
    r"\bpython\b",r"\bjava\b",r"\bjavascript\b",r"\btypescript\b",r"\bc\+\+\b",
    r"\bc#\b",r"\brust\b",r"\bgo(?:lang)?\b",r"\bkotlin\b",r"\bswift\b",
    r"\bscala\b",r"\bphp\b",r"\bruby\b",r"\bmatlab\b",r"\bperl\b",
    r"\bbash\b",r"\bshell\b",r"\bsql\b",r"\bnosql\b",r"\bhtml\b",r"\bcss\b",
    r"\breact(?:\.js)?\b",r"\bvue(?:\.js)?\b",r"\bangular\b",r"\bnext(?:\.js)?\b",
    r"\bnuxt(?:\.js)?\b",r"\bsvelte\b",r"\bdjango\b",r"\bflask\b",r"\bfastapi\b",
    r"\bspring(?:\s*boot)?\b",r"\bnode(?:\.js)?\b",r"\bexpress(?:\.js)?\b",
    r"\blaravel\b",r"\brails\b",r"\bsymfony\b",r"\bstreamlit\b",
    r"\bpytorch\b",r"\btensorflow\b",r"\bkeras\b",r"\bscikit[-\s]?learn\b",
    r"\bhugging\s*face\b",r"\bpandas\b",r"\bnumpy\b",r"\bscipy\b",
    r"\bmatplotlib\b",r"\bseaborn\b",r"\bplotly\b",r"\blangchain\b",
    r"\bmysql\b",r"\bpostgres(?:ql)?\b",r"\bmongodb\b",r"\bredis\b",
    r"\belasticsearch\b",r"\bcassandra\b",r"\bsqlite\b",r"\boracle\b",
    r"\bdynamodb\b",r"\bcosmosdb\b",r"\bfirebase\b",r"\bsupabase\b",
    r"\bsnowflake\b",r"\bbigquery\b",r"\bdatabricks\b",r"\bneo4j\b",
    r"\baws\b",r"\bazure\b",r"\bgcp\b",r"\bgoogle\s+cloud\b",
    r"\bdocker\b",r"\bkubernetes\b",r"\bk8s\b",r"\bterraform\b",
    r"\bansible\b",r"\bjenkins\b",r"\bgithub\s+actions\b",r"\bcircle\s*ci\b",
    r"\bhelm\b",r"\bargocd\b",r"\bci/cd\b",r"\bdevsecops\b",
    r"\blambda\b",r"\bec2\b",r"\bs3\b",r"\brds\b",r"\bvpc\b",
    r"\bmachine\s+learning\b",r"\bdeep\s+learning\b",r"\bnatural\s+language\s+processing\b",
    r"\bnlp\b",r"\bcomputer\s+vision\b",r"\bllm\b",r"\bgpt\b",r"\bbert\b",
    r"\btransformers\b",r"\bmlops\b",r"\bdata\s+science\b",r"\bdata\s+engineering\b",
    r"\bapache\s+spark\b",r"\bhadoop\b",r"\bkafka\b",r"\bairflow\b",
    r"\bdbt\b",r"\bflink\b",r"\bpower\s*bi\b",r"\btableau\b",r"\blooker\b",
    r"\bqlik\b",r"\bexcel\b",r"\bsas\b",r"\bspss\b",
    r"\bagile\b",r"\bscrum\b",r"\bkanban\b",r"\bdevops\b",r"\bsre\b",
    r"\bmicroservices\b",r"\brestful?\b",r"\bgraphql\b",r"\bgrpc\b",
    r"\bsoa\b",r"\bcybersecurity\b",r"\bpenetration\s+testing\b",r"\bsiem\b",
    r"\bleadership\b",r"\bcommunication\b",r"\bteamwork\b",r"\bproblem[-\s]solving\b",
    r"\bproject\s+management\b",r"\bstakeholder\b",r"\bmentoring\b",
]
COMPILED_TECH = [re.compile(p, re.I) for p in TECH_PATTERNS]

EMAIL_RE    = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE    = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-/]+", re.I)
GITHUB_RE   = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w\-/]+", re.I)

# ─── Cached Model Loaders ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP model…")
def load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Add it to requirements.txt: "
            "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
        )


@st.cache_resource(show_spinner="Loading Sentence Transformer…")
def load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def encode_text_cached(text: str, text_hash: str):
    """Cache sentence embeddings keyed by content hash."""
    model = load_sentence_transformer()
    import torch
    with torch.no_grad():
        emb = model.encode(text[:3000], convert_to_numpy=True)
    return emb


# ─── File Extraction ──────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        pages = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if len(pdf.pages) == 0:
                raise ValueError("PDF has no pages.")
            for page in pdf.pages:
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append(text)
                except Exception:
                    continue
        if not pages:
            raise ValueError("No readable text found in PDF.")
        return "\n".join(pages)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"PDF extraction failed: {e}")


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        tables_text = []
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        tables_text.append(cell.text.strip())
        result = "\n".join(paragraphs + tables_text)
        if not result.strip():
            raise ValueError("DOCX file appears to be empty.")
        return result
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"DOCX extraction failed: {e}")


def extract_text(uploaded_file) -> str:
    try:
        raw = uploaded_file.read()
    except Exception as e:
        raise ValueError(f"Cannot read file '{uploaded_file.name}': {e}")
    if not raw or len(raw) < 10:
        raise ValueError(f"File '{uploaded_file.name}' is empty or too small.")
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(raw)
    elif name.endswith(".docx"):
        return extract_text_from_docx(raw)
    elif name.endswith(".txt"):
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                text = raw.decode(enc)
                if text.strip():
                    return text
            except Exception:
                continue
        raise ValueError(f"Cannot decode TXT file '{uploaded_file.name}'.")
    else:
        raise ValueError(f"Unsupported format: '{uploaded_file.name}'. Use PDF, DOCX, or TXT.")


# ─── Date / Experience Parsing ────────────────────────────────────────────────────
MONTH_MAP = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
}
PRESENT_WORDS = {"present","current","now","till date","to date","ongoing"}

def _parse_month_year(month_str: str, year_str: str) -> date:
    m = MONTH_MAP.get(month_str[:3].lower(), 1)
    return date(int(year_str), m, 1)


def calculate_experience_from_dates(text: str) -> float:
    """
    Supports:
      - Jan 2020 – Dec 2022
      - January 2020 to Present
      - 2020 – 2023
      - 2020 - Present
      - 01/2020 – 12/2022  (MM/YYYY)
      - 2020-Present  (no spaces)
    """
    segments: list[tuple[date, date]] = []
    today = date.today()

    # Pattern A: Month YYYY [–] Month/Present YYYY?
    pat_a = re.compile(
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"[\s,\.\-]*(\d{4})\s*[\-–—to/]+\s*"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
        r"present|current|now|till\s*date|to\s*date|ongoing)"
        r"(?:[\s,\.]*(\d{4}))?",
        re.I,
    )
    for m in pat_a.finditer(text):
        try:
            start = _parse_month_year(m.group(1), m.group(2))
            end_word = m.group(3).lower().strip()
            if any(p in end_word for p in PRESENT_WORDS):
                end = today
            else:
                if m.group(4):
                    end = _parse_month_year(m.group(3), m.group(4))
                else:
                    continue
            if start <= end:
                segments.append((start, end))
        except Exception:
            continue

    # Pattern B: MM/YYYY – MM/YYYY or MM/YYYY – Present
    pat_b = re.compile(
        r"(0?[1-9]|1[0-2])[/\-](20\d{2}|19\d{2})\s*[\-–—to]+\s*"
        r"(?:(0?[1-9]|1[0-2])[/\-](20\d{2}|19\d{2})|(present|current|now|ongoing))",
        re.I,
    )
    for m in pat_b.finditer(text):
        try:
            start = date(int(m.group(2)), int(m.group(1)), 1)
            if m.group(5):
                end = today
            else:
                end = date(int(m.group(4)), int(m.group(3)), 1)
            if start <= end:
                segments.append((start, end))
        except Exception:
            continue

    # Pattern C: YYYY – YYYY or YYYY – Present (bare year ranges)
    pat_c = re.compile(
        r"(?<!\d)(20\d{2}|19\d{2})\s*[\-–—]+\s*(20\d{2}|19\d{2}|present|current|now|ongoing)(?!\d)",
        re.I,
    )
    for m in pat_c.finditer(text):
        try:
            start = date(int(m.group(1)), 1, 1)
            end_str = m.group(2).lower()
            if any(p in end_str for p in PRESENT_WORDS):
                end = today
            else:
                end = date(int(m.group(2)), 12, 31)
            if start <= end:
                segments.append((start, end))
        except Exception:
            continue

    if not segments:
        # Fallback: explicit "X years experience" mention
        yr_mentions = re.findall(
            r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:relevant\s+|total\s+)?experience",
            text, re.I
        )
        if yr_mentions:
            return max(float(y) for y in yr_mentions)
        return 0.0

    # Deduplicate and merge overlapping segments
    segments = list(set(segments))
    segments.sort(key=lambda x: x[0])
    merged: list[tuple[date, date]] = [segments[0]]
    for s, e in segments[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    total_days = sum((e - s).days for s, e in merged)
    return round(total_days / 365.25, 1)


# ─── NLP / Skill Extraction ───────────────────────────────────────────────────────
def extract_skills_nlp(text: str, nlp) -> list[str]:
    found: set[str] = set()

    # Regex patterns (fast, runs on full text)
    for pattern in COMPILED_TECH:
        for m in pattern.finditer(text):
            skill = re.sub(r"\s+", " ", m.group(0).strip()).lower()
            found.add(skill)

    # spaCy on first 15 000 chars only (en_core_web_sm is lighter)
    doc = nlp(text[:15000])
    for ent in doc.ents:
        if ent.label_ in ("PRODUCT", "ORG"):
            tok = ent.text.strip()
            if 2 <= len(tok) <= 40 and not tok.isnumeric():
                found.add(tok.lower())

    for chunk in doc.noun_chunks:
        c = chunk.text.strip().lower()
        if 3 <= len(c) <= 35 and any(p.search(c) for p in COMPILED_TECH):
            found.add(c)

    skills = []
    for s in found:
        s = s.strip(" .,;:()")
        s = re.sub(r"[^\w\s\+\#\/\-\.]", "", s).strip()
        if s and len(s) >= 2:
            skills.append(s)
    return list(set(skills))


def extract_name_spacy(text: str, nlp) -> str:
    doc = nlp(text[:400])
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
            return ent.text.strip()
    for line in text.splitlines()[:8]:
        line = line.strip()
        if 2 <= len(line.split()) <= 5 and re.match(r"^[A-Za-z\s\.\-]+$", line):
            return line
    return "Unknown"


def extract_education(text: str) -> list[str]:
    educations = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        ll = line.lower()
        if any(kw in ll for kw in DEGREE_KEYWORDS):
            ctx = " ".join(lines[max(0, i-1):i+2]).strip()
            if ctx and len(ctx) > 8:
                educations.append(ctx[:200])
    return list(dict.fromkeys(educations))[:5]


def extract_certifications(text: str) -> list[str]:
    certs = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        ll = line.lower()
        if any(kw in ll for kw in CERT_KEYWORDS):
            ctx = " ".join(lines[max(0, i-1):i+2]).strip()
            if ctx:
                certs.append(ctx[:200])
    return list(dict.fromkeys(certs))[:8]


def extract_companies_and_designations(text: str, nlp) -> tuple[list[str], list[str]]:
    doc = nlp(text[:12000])  # sm model: limit input
    companies = list(dict.fromkeys(ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"))[:8]
    desig_pat = re.compile(
        r"(senior|junior|lead|principal|staff|chief|head|vp|vice\s+president|director|manager|"
        r"engineer|developer|architect|analyst|scientist|designer|consultant|specialist|"
        r"associate|intern|executive|officer|president|cto|ceo|cfo|coo)\s+[\w\s]{2,40}",
        re.I,
    )
    designations = []
    for m in desig_pat.finditer(text):
        d = m.group(0).strip()
        if 3 <= len(d.split()) <= 6:
            designations.append(d.title())
    return companies, list(dict.fromkeys(designations))[:6]


def extract_location(text: str, nlp) -> str:
    doc = nlp(text[:1500])
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            return ent.text.strip()
    return ""


def extract_projects(text: str) -> list[str]:
    projects = []
    lines = text.splitlines()
    in_section = False
    for line in lines:
        ll = line.lower().strip()
        if re.match(r"^projects?[\s:]*$", ll):
            in_section = True
            continue
        if in_section:
            if re.match(r"^(experience|education|skills|certif|work|employment)", ll):
                in_section = False
                continue
            if len(line.strip()) > 10:
                projects.append(line.strip()[:200])
    return projects[:6]


# ─── Parsing (cached) ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def parse_resume(_text: str, _text_hash: str) -> dict:
    nlp = load_spacy()
    text = _text
    name      = extract_name_spacy(text, nlp)
    email     = next(iter(EMAIL_RE.findall(text)), "")
    phones    = PHONE_RE.findall(text)
    phone     = phones[0].strip() if phones else ""
    linkedin  = next(iter(LINKEDIN_RE.findall(text)), "")
    github    = next(iter(GITHUB_RE.findall(text)), "")
    location  = extract_location(text, nlp)
    education = extract_education(text)
    certs     = extract_certifications(text)
    projects  = extract_projects(text)
    companies, designations = extract_companies_and_designations(text, nlp)
    skills    = extract_skills_nlp(text, nlp)
    exp_years = calculate_experience_from_dates(text)
    return {
        "name": name, "email": email, "phone": phone,
        "linkedin": linkedin, "github": github, "location": location,
        "education": education, "certifications": certs, "projects": projects,
        "companies": companies, "designations": designations,
        "skills": skills, "experience_years": exp_years, "raw_text": text,
    }


@st.cache_data(show_spinner=False)
def parse_job_description(_text: str, _hash: str) -> dict:

    if _text is None:
        raise ValueError("JD text is None")

    text = str(_text or "").strip()

    if not text:
        raise ValueError("JD text is empty")

    nlp = load_spacy()
    nlp  = load_spacy()
    text = _text
    skills = extract_skills_nlp(text, nlp)
    exp_match  = re.search(r"(\d+)\+?\s*(?:to\s*\d+)?\s*years?\s+(?:of\s+)?(?:relevant\s+)?experience", text, re.I)
    required_exp = float(exp_match.group(1)) if exp_match else 0.0
    edu_reqs = list(set(deg.upper() for deg in DEGREE_KEYWORDS if deg in text.lower()))[:6]
    role_match = re.search(r"(?:job\s+title|position|role)[:\s]+([^\n]{5,60})", text, re.I)
    role = role_match.group(1).strip() if role_match else "Position"

    def extract_section(pattern: str) -> str:
        m = re.search(
            pattern + r"[:\s\n]+([\s\S]{30,800}?)(?=\n[A-Z][A-Za-z ]+:|$)",
            text,
            re.I
        )
        
        if not m:
            return ""
            
        section = m.group(1)
        
        if section is None:
            return ""
            
        return str(section).strip()

    return {
        "role": role,
        "required_skills": skills,
        "required_experience_years": required_exp,
        "required_education": edu_reqs,
        "responsibilities": extract_section(r"responsibilit(?:ies|y)|duties|what\s+you.ll\s+do"),
        "requirements":     extract_section(r"requirements?|qualifications?|what\s+we.re\s+looking"),
        "nice_to_have":     extract_section(r"nice[\s\-]to[\s\-]have|preferred|bonus|plus"),
        "raw_text": text,
    }


# ─── Matching Engine ──────────────────────────────────────────────────────────────
def compute_skill_match(
    resume_skills: list[str],
    jd_skills: list[str],
    critical_skills: list[str],
) -> tuple[float, list, list, list, list]:
    from rapidfuzz import fuzz
    if not jd_skills:
        return 0.0, [], [], resume_skills[:], []

    threshold = 82
    matched, missing = [], []
    for jd_skill in jd_skills:
        best = max(
            (fuzz.token_sort_ratio(jd_skill.lower(), r.lower()) for r in resume_skills),
            default=0,
        )
        (matched if best >= threshold else missing).append(jd_skill)

    additional = [
        r for r in resume_skills
        if max((fuzz.token_sort_ratio(r.lower(), jd.lower()) for jd in jd_skills), default=0) < threshold
    ]

    # Critical skills that are missing
    crit_lower = {c.lower().strip() for c in critical_skills}
    critical_missing = [
        s for s in missing
        if any(fuzz.token_sort_ratio(s.lower(), c) >= threshold for c in crit_lower)
    ]

    # Apply penalty for missing critical skills (−5 pts each, max −25)
    base_score = len(matched) / len(jd_skills)
    penalty = min(0.25, len(critical_missing) * 0.05)
    skill_score = max(0.0, base_score - penalty)

    return skill_score, matched, missing, additional, critical_missing


def compute_semantic_match(resume_text: str, jd_text: str, jd_hash: str, resume_hash: str) -> float:
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        emb_r = encode_text_cached(resume_text[:3000], resume_hash)
        emb_j = encode_text_cached(jd_text[:3000],     jd_hash)
        sim = float(cosine_similarity([emb_r], [emb_j])[0][0])
        return max(0.0, min(1.0, sim))
    except Exception:
        return 0.0


def compute_experience_match(resume_exp: float, required_exp: float) -> float:
    if required_exp <= 0:
        return 1.0 if resume_exp > 0 else 0.5
    return min(1.0, round(resume_exp / required_exp, 3))


def compute_education_match(resume_education: list[str], jd_education: list[str]) -> float:
    if not jd_education:
        return 1.0
    edu_text = " ".join(resume_education).lower()
    matched = sum(1 for req in jd_education if req.lower() in edu_text)
    return matched / len(jd_education)


def compute_ats_score(resume: dict, jd: dict, skill_match: float) -> int:
    score = 0
    jd_words     = set(re.findall(r"\b\w{4,}\b", jd["raw_text"].lower()))
    resume_words = set(re.findall(r"\b\w{4,}\b", resume["raw_text"].lower()))
    overlap = len(jd_words & resume_words) / max(len(jd_words), 1)
    score += min(30, int(overlap * 60))
    if resume["email"]:    score += 8
    if resume["phone"]:    score += 6
    if resume["linkedin"]: score += 6
    score += int(skill_match * 25)
    req_exp = jd.get("required_experience_years", 0)
    if req_exp > 0:
        score += int(min(1.0, resume["experience_years"] / req_exp) * 15)
    else:
        score += 10 if resume["experience_years"] > 0 else 5
    edu_text = " ".join(resume["education"]).lower()
    for deg in ["bachelor","master","phd","b.tech","m.tech","mba"]:
        if deg in edu_text:
            score += 10
            break
    return min(100, max(0, score))


def compute_overall_score(components: dict) -> float:
    return round(
        components["skill_match"]      * WEIGHTS["skill_match"]
        + components["semantic_match"] * WEIGHTS["semantic_match"]
        + components["experience_match"] * WEIGHTS["experience_match"]
        + (components["ats_score"] / 100) * WEIGHTS["ats_score"]
        + components["education_match"] * WEIGHTS["education_match"],
        4,
    )


def get_fit_badge(score_pct: float) -> tuple[str, str]:
    if score_pct >= 80: return "Excellent Fit",  "badge-excellent"
    if score_pct >= 65: return "Strong Match",   "badge-strong"
    if score_pct >= 50: return "Good Match",      "badge-good"
    if score_pct >= 35: return "Average Match",   "badge-average"
    return "Poor Match", "badge-poor"


def generate_ats_suggestions(resume: dict, jd: dict, match_result: dict) -> list[str]:
    suggestions = []
    if not resume["email"]:
        suggestions.append("Add a professional email address — ATS systems require it for contact matching.")
    if not resume["phone"]:
        suggestions.append("Include a phone number to complete contact info (adds ~6 ATS pts).")
    if not resume["linkedin"]:
        suggestions.append("Add your LinkedIn URL — increases ATS contact score by 6 pts.")
    if match_result["components"]["skill_match"] < 0.5:
        suggestions.append("Mirror more keywords from the JD directly in your Skills section.")
    if match_result["ats_score"] < 60:
        suggestions.append("Increase keyword overlap with the JD — use the exact same terminology for tools and technologies.")
    if resume["experience_years"] < jd.get("required_experience_years", 0):
        suggestions.append(f"JD requires {jd['required_experience_years']} yrs; highlight all relevant roles with explicit date ranges.")
    if not resume["education"]:
        suggestions.append("Add an Education section — degree keywords boost ATS education scoring.")
    if match_result["missing_skills"]:
        top_missing = ", ".join(match_result["missing_skills"][:5])
        suggestions.append(f"Add these JD keywords to your resume if applicable: {top_missing}.")
    if not suggestions:
        suggestions.append("ATS profile looks strong. Keep resume formatting clean (no tables/images in Skills section).")
    return suggestions


def generate_resume_suggestions(resume: dict, jd: dict, match_result: dict) -> list[str]:
    suggestions = []
    if not resume["github"]:
        suggestions.append("Add a GitHub profile link to showcase your portfolio and projects.")
    if not resume["projects"]:
        suggestions.append("Include a Projects section with 2–3 relevant technical projects.")
    if not resume["certifications"]:
        suggestions.append(f"Consider adding certifications relevant to the role ({jd['role']}).")
    if match_result["missing_skills"]:
        suggestions.append(f"Skill gap: consider learning or highlighting — {', '.join(match_result['missing_skills'][:4])}.")
    if resume["experience_years"] == 0:
        suggestions.append("Ensure work history includes explicit date ranges (e.g. Jan 2020 – Present).")
    if len(resume["skills"]) < 8:
        suggestions.append("Expand your Skills section — aim for 10–15 relevant technical skills.")
    if not resume["designations"]:
        suggestions.append("Add your job titles/designations clearly (e.g. 'Senior Software Engineer').")
    if match_result["components"]["semantic_match"] < 0.4:
        suggestions.append("Rephrase your summary/objective to better align with the job description language.")
    return suggestions[:6]


@st.cache_data(show_spinner=False)
def match_resume_to_jd(
    _resume: dict, _jd: dict, cache_key: str, critical_skills_key: str
) -> dict:
    critical_skills = st.session_state.get("critical_skills", [])
    skill_score, matched, missing, additional, crit_missing = compute_skill_match(
        _resume["skills"], _jd["required_skills"], critical_skills
    )
    res_hash = hashlib.md5(_resume["raw_text"][:3000].encode()).hexdigest()
    jd_hash  = hashlib.md5(_jd["raw_text"][:3000].encode()).hexdigest()
    semantic_score  = compute_semantic_match(_resume["raw_text"], _jd["raw_text"], jd_hash, res_hash)
    exp_score       = compute_experience_match(_resume["experience_years"], _jd["required_experience_years"])
    edu_score       = compute_education_match(_resume["education"], _jd["required_education"])
    ats             = compute_ats_score(_resume, _jd, skill_score)

    components = {
        "skill_match": skill_score, "semantic_match": semantic_score,
        "experience_match": exp_score, "ats_score": ats, "education_match": edu_score,
    }
    overall = compute_overall_score(components)
    return {
        "components": components,
        "overall_score": overall,
        "matched_skills": matched,
        "missing_skills": missing,
        "additional_skills": additional,
        "critical_missing": crit_missing,
        "ats_score": ats,
    }


# ─── Gemini AI Analysis ───────────────────────────────────────────────────────────
def validate_gemini_key(api_key: str) -> tuple[bool, str]:
    if not api_key or len(api_key) < 20:
        return False, "API key is too short or empty."
    if not api_key.startswith("AI"):
        return False, "Gemini API keys typically start with 'AI'."
    return True, ""


def gemini_analyze(resume: dict, jd: dict, match_result: dict, api_key: str) -> dict:
    valid, err = validate_gemini_key(api_key)
    if not valid:
        return {"error": f"Invalid API key: {err}"}
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""You are a senior technical recruiter. Analyze this candidate vs the job description.
Return ONLY a valid JSON object — no markdown, no backticks, no explanation.

CANDIDATE:
Name: {resume['name']}
Experience: {resume['experience_years']} years
Skills: {', '.join(resume['skills'][:30])}
Education: {'; '.join(resume['education'][:3])}
Companies: {', '.join(resume['companies'][:5])}
Designations: {', '.join(resume['designations'][:3])}
Certifications: {'; '.join(resume['certifications'][:3])}

JOB:
Role: {jd['role']}
Required Skills: {', '.join(jd['required_skills'][:20])}
Required Experience: {jd['required_experience_years']} years
Requirements: {jd['requirements'][:500]}

SCORES:
Overall: {round(match_result['overall_score']*100,1)}%
Skill: {round(match_result['components']['skill_match']*100,1)}%
Semantic: {round(match_result['components']['semantic_match']*100,1)}%
ATS: {match_result['ats_score']}/100
Matched: {', '.join(match_result['matched_skills'][:12])}
Missing: {', '.join(match_result['missing_skills'][:12])}
Critical Missing: {', '.join(match_result.get('critical_missing',[])[:8])}

Return exactly:
{{
  "candidate_summary": "2-3 sentence professional summary",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "missing_skills_analysis": ["gap 1 with context", "gap 2 with context"],
  "interview_recommendation": "Strongly Recommend / Recommend / Consider / Do Not Recommend",
  "hiring_decision": "Hire / Shortlist / Reject",
  "confidence_level": "High / Medium / Low",
  "improvement_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"],
  "cultural_fit_score": 75,
  "technical_depth": "Expert / Senior / Mid-level / Junior",
  "red_flags": [],
  "positive_indicators": ["indicator 1", "indicator 2"]
}}"""

        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        # Try to extract JSON object if extra text present
        json_match = re.search(r"\{[\s\S]+\}", raw)
        if json_match:
            raw = json_match.group(0)
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return {"error": f"Gemini returned non-JSON response: {str(e)[:100]}"}
    except Exception as e:
        err_str = str(e)
        if "API_KEY" in err_str or "401" in err_str or "403" in err_str:
            return {"error": "Invalid or unauthorized Gemini API key. Check your key at aistudio.google.com."}
        if "quota" in err_str.lower() or "429" in err_str:
            return {"error": "Gemini API quota exceeded. Try again later or upgrade your plan."}
        return {"error": f"Gemini analysis failed: {err_str[:200]}"}


# ─── Excel Export ─────────────────────────────────────────────────────────────────
def build_excel_report(all_results: list[dict], jd: dict) -> bytes:
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:

        # ── Sheet 1: Executive Summary ───────────────────────────────────────────
        scores   = [r["match_result"]["overall_score"]*100 for r in all_results]
        ats_vals = [r["match_result"]["ats_score"] for r in all_results]
        strong   = sum(1 for s in scores if s >= 65)
        medium   = sum(1 for s in scores if 40 <= s < 65)
        weak     = sum(1 for s in scores if s < 40)
        top      = all_results[0]["resume"]["name"] if all_results else "N/A"

        exec_data = {
            "Metric": [
                "Total Candidates", "Strong Matches (≥65%)", "Medium Matches (40-64%)",
                "Weak Matches (<40%)", "Average Match Score", "Average ATS Score",
                "Top Candidate", "JD Role", "Required Experience",
            ],
            "Value": [
                len(all_results), strong, medium, weak,
                f"{np.mean(scores):.1f}%" if scores else "N/A",
                f"{np.mean(ats_vals):.0f}/100" if ats_vals else "N/A",
                top, jd["role"], f"{jd['required_experience_years']} years",
            ],
        }
        pd.DataFrame(exec_data).to_excel(writer, sheet_name="Executive Summary", index=False)

        # ── Sheet 2: Ranking ─────────────────────────────────────────────────────
        rank_rows = []
        for r in all_results:
            mr = r["match_result"]
            ai = r.get("ai_analysis", {})
            badge, _ = get_fit_badge(mr["overall_score"]*100)
            rank_rows.append({
                "Rank": r["rank"],
                "Name": r["resume"]["name"],
                "Fit Badge": badge,
                "Overall %": round(mr["overall_score"]*100, 1),
                "Skill Match %": round(mr["components"]["skill_match"]*100, 1),
                "Semantic %": round(mr["components"]["semantic_match"]*100, 1),
                "Exp Match %": round(mr["components"]["experience_match"]*100, 1),
                "ATS Score": mr["ats_score"],
                "Edu Match %": round(mr["components"]["education_match"]*100, 1),
                "Experience (Yrs)": r["resume"]["experience_years"],
                "Email": r["resume"]["email"],
                "Phone": r["resume"]["phone"],
                "Location": r["resume"]["location"],
                "Hiring Decision": ai.get("hiring_decision", ""),
                "Interview Rec": ai.get("interview_recommendation", ""),
            })
        pd.DataFrame(rank_rows).to_excel(writer, sheet_name="Ranking", index=False)

        # ── Sheet 3: Skill Matrix ─────────────────────────────────────────────────
        jd_skills_top = jd["required_skills"][:25]
        matrix_rows = []
        for r in all_results:
            matched_set = set(s.lower() for s in r["match_result"]["matched_skills"])
            row = {"Candidate": r["resume"]["name"]}
            for skill in jd_skills_top:
                row[skill] = "✓" if skill.lower() in matched_set else "✗"
            matrix_rows.append(row)
        pd.DataFrame(matrix_rows).to_excel(writer, sheet_name="Skill Matrix", index=False)

        # ── Sheet 4: ATS Analysis ─────────────────────────────────────────────────
        ats_rows = []
        for r in all_results:
            mr = r["match_result"]
            ats_suggestions = generate_ats_suggestions(r["resume"], jd, mr)
            ats_rows.append({
                "Name": r["resume"]["name"],
                "ATS Score": mr["ats_score"],
                "Has Email": "Yes" if r["resume"]["email"] else "No",
                "Has Phone": "Yes" if r["resume"]["phone"] else "No",
                "Has LinkedIn": "Yes" if r["resume"]["linkedin"] else "No",
                "Has GitHub": "Yes" if r["resume"]["github"] else "No",
                "ATS Suggestions": " | ".join(ats_suggestions[:3]),
            })
        pd.DataFrame(ats_rows).to_excel(writer, sheet_name="ATS Analysis", index=False)

        # ── Sheet 5: Missing Skills ───────────────────────────────────────────────
        missing_rows = []
        for r in all_results:
            mr = r["match_result"]
            missing_rows.append({
                "Name": r["resume"]["name"],
                "Missing Skills": "; ".join(mr["missing_skills"]),
                "Critical Missing": "; ".join(mr.get("critical_missing", [])),
                "Matched Skills": "; ".join(mr["matched_skills"]),
                "Additional Skills": "; ".join(mr["additional_skills"][:10]),
                "Missing Count": len(mr["missing_skills"]),
                "Matched Count": len(mr["matched_skills"]),
            })
        pd.DataFrame(missing_rows).to_excel(writer, sheet_name="Missing Skills", index=False)

        # ── Sheet 6: Gemini Insights ──────────────────────────────────────────────
        gemini_rows = []
        for r in all_results:
            ai = r.get("ai_analysis", {})
            if ai and "error" not in ai:
                gemini_rows.append({
                    "Name": r["resume"]["name"],
                    "Summary": ai.get("candidate_summary", ""),
                    "Strengths": " | ".join(ai.get("strengths", [])),
                    "Weaknesses": " | ".join(ai.get("weaknesses", [])),
                    "Missing Skills Analysis": " | ".join(ai.get("missing_skills_analysis", [])),
                    "Improvement Suggestions": " | ".join(ai.get("improvement_suggestions", [])),
                    "Hiring Decision": ai.get("hiring_decision", ""),
                    "Interview Rec": ai.get("interview_recommendation", ""),
                    "Technical Depth": ai.get("technical_depth", ""),
                    "Confidence": ai.get("confidence_level", ""),
                    "Red Flags": " | ".join(ai.get("red_flags", [])),
                    "Positive Indicators": " | ".join(ai.get("positive_indicators", [])),
                })
        if gemini_rows:
            pd.DataFrame(gemini_rows).to_excel(writer, sheet_name="Gemini Insights", index=False)

    return output.getvalue()


# ─── Visualizations ───────────────────────────────────────────────────────────────
def fig_defaults(fig):
    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=CHART_BG,
        font=dict(color=FONT_COLOR, family="Inter"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def chart_match_scores(all_results: list) -> go.Figure:
    names  = [r["resume"]["name"].split()[-1] for r in all_results]
    scores = [round(r["match_result"]["overall_score"]*100, 1) for r in all_results]
    colors = ["#34d399" if s>=65 else "#fbbf24" if s>=40 else "#f87171" for s in scores]
    fig = go.Figure(go.Bar(
        x=names, y=scores, marker_color=colors,
        text=[f"{s}%" for s in scores], textposition="outside",
        textfont=dict(color=FONT_COLOR, size=11),
    ))
    fig.update_layout(
        title="Overall Match Score",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, title="Match %", range=[0,115]),
    )
    return fig_defaults(fig)


def chart_ats_scores(all_results: list) -> go.Figure:
    names = [r["resume"]["name"].split()[-1] for r in all_results]
    ats   = [r["match_result"]["ats_score"] for r in all_results]
    fig = go.Figure(go.Bar(
        x=names, y=ats, marker_color=CHART_COLORS[1],
        text=ats, textposition="outside",
        textfont=dict(color=FONT_COLOR, size=11),
    ))
    fig.update_layout(
        title="ATS Score",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, title="ATS Score", range=[0,115]),
    )
    return fig_defaults(fig)


def chart_experience_distribution(all_results: list) -> go.Figure:
    exps = [r["resume"]["experience_years"] for r in all_results]
    fig = go.Figure(go.Histogram(
        x=exps, nbinsx=10, marker_color=CHART_COLORS[2],
        marker_line_color="rgba(255,255,255,0.1)", marker_line_width=1,
    ))
    fig.update_layout(
        title="Experience Distribution (Years)",
        xaxis=dict(gridcolor=GRID_COLOR, title="Years"),
        yaxis=dict(gridcolor=GRID_COLOR, title="Candidates"),
    )
    return fig_defaults(fig)


def chart_skill_frequency(all_results: list, jd: dict) -> go.Figure:
    all_skills = []
    for r in all_results:
        all_skills.extend(r["resume"]["skills"])
    counts = Counter(all_skills).most_common(20)
    if not counts:
        return fig_defaults(go.Figure())
    labels = [c[0][:20] for c in counts]
    values = [c[1] for c in counts]
    jd_set = {s.lower() for s in jd["required_skills"]}
    colors = ["#34d399" if l.lower() in jd_set else "#6366f1" for l in labels]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=values, textposition="outside",
        textfont=dict(color=FONT_COLOR, size=10),
    ))
    fig.update_layout(
        title="Top Skills Frequency (green=JD requirement)",
        xaxis=dict(gridcolor=GRID_COLOR, title="Count"),
        yaxis=dict(autorange="reversed", gridcolor=GRID_COLOR),
        height=500,
    )
    return fig_defaults(fig)


def chart_radar(results_subset: list) -> go.Figure:
    categories = ["Skill Match","Semantic","Experience","ATS","Education"]
    fig = go.Figure()
    for i, r in enumerate(results_subset):
        c = r["match_result"]["components"]
        vals = [
            c["skill_match"]*100, c["semantic_match"]*100,
            c["experience_match"]*100, r["match_result"]["ats_score"],
            c["education_match"]*100,
        ]
        vals_c = vals + [vals[0]]
        cats_c = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_c, theta=cats_c, fill="toself",
            name=r["resume"]["name"].split()[0],
            line_color=CHART_COLORS[i % len(CHART_COLORS)],
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(30,34,53,0.8)",
            radialaxis=dict(visible=True, range=[0,100], gridcolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR)),
            angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR)),
        ),
        title="Candidate Radar Comparison (Top 5)",
        showlegend=True,
        legend=dict(bgcolor="rgba(30,34,53,0.8)"),
    )
    return fig_defaults(fig)


def chart_score_breakdown(match_result: dict) -> go.Figure:
    c = match_result["components"]
    categories = ["Skill Match","Semantic Match","Exp Match","Edu Match"]
    raw    = [c["skill_match"]*100, c["semantic_match"]*100, c["experience_match"]*100, c["education_match"]*100]
    weighted = [
        c["skill_match"]*WEIGHTS["skill_match"]*100,
        c["semantic_match"]*WEIGHTS["semantic_match"]*100,
        c["experience_match"]*WEIGHTS["experience_match"]*100,
        c["education_match"]*WEIGHTS["education_match"]*100,
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Raw Score", x=categories, y=raw, marker_color=CHART_COLORS[0]))
    fig.add_trace(go.Bar(name="Weighted", x=categories, y=weighted, marker_color=CHART_COLORS[2]))
    fig.update_layout(
        title="Score Breakdown",
        barmode="group",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, title="Score %"),
    )
    return fig_defaults(fig)


def chart_skill_heatmap(all_results: list, jd: dict) -> go.Figure:
    jd_skills = jd["required_skills"][:20]
    if not jd_skills or not all_results:
        return fig_defaults(go.Figure())
    from rapidfuzz import fuzz
    threshold = 82
    names = [r["resume"]["name"].split()[0] for r in all_results]
    matrix = []
    for skill in jd_skills:
        row = []
        for r in all_results:
            matched = max(
                (fuzz.token_sort_ratio(skill.lower(), s.lower()) for s in r["resume"]["skills"]),
                default=0,
            )
            row.append(1 if matched >= threshold else 0)
        matrix.append(row)
    fig = go.Figure(go.Heatmap(
        z=matrix, x=names, y=[s[:25] for s in jd_skills],
        colorscale=[[0,"#1e2235"],[0.5,"#6366f1"],[1,"#34d399"]],
        showscale=True, colorbar=dict(tickvals=[0,1], ticktext=["Missing","Present"]),
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Skill Heatmap: Candidates vs JD Requirements",
        xaxis=dict(title="Candidates", tickangle=-30),
        yaxis=dict(title="Required Skills", autorange="reversed"),
        height=max(350, len(jd_skills)*28),
    )
    return fig_defaults(fig)


# ─── UI Helpers ───────────────────────────────────────────────────────────────────
def render_skill_tags(skills: list[str], tag_class: str, label: str, limit: int = 25):
    if not skills:
        return
    tags = " ".join(f'<span class="skill-tag {tag_class}">{s}</span>' for s in skills[:limit])
    st.markdown(f"**{label}:** {tags}", unsafe_allow_html=True)


def render_kpi_row(kpis: list[tuple]):
    cols = st.columns(len(kpis))
    for col, (label, value, delta) in zip(cols, kpis):
        with col:
            delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-value">{value}</div>'
                f'<div class="kpi-label">{label}</div>{delta_html}</div>',
                unsafe_allow_html=True,
            )


def candidate_search(all_results: list, query: str) -> list[dict]:
    if not query.strip():
        return all_results
    q = query.lower().strip()
    filtered = []
    for r in all_results:
        res = r["resume"]
        searchable = " ".join([
            res["name"], res["email"], res["location"],
            " ".join(res["companies"]), " ".join(res["skills"]),
        ]).lower()
        if q in searchable:
            filtered.append(r)
    return filtered


# ─── Main App ─────────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:12px 0;">
          <div style="font-size:2rem;">🎯</div>
          <div style="font-size:1.1rem;font-weight:700;color:#818cf8;">ATS Platform v2</div>
          <div style="font-size:0.75rem;color:#475569;margin-top:2px;">AI Resume Matcher</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        gemini_api_key = st.text_input(
            "🔑 Gemini API Key", type="password", placeholder="AIza…",
            help="Get your key at https://aistudio.google.com/",
        )
        if gemini_api_key:
            valid, err = validate_gemini_key(gemini_api_key)
            if not valid:
                st.warning(f"⚠️ {err}")
            else:
                st.success("✅ Key format valid")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("**⚙️ Matching Weights**")
        with st.expander("Adjust Weights", expanded=False):
            w_skill = st.slider("Skill Match",    0.0, 1.0, WEIGHTS["skill_match"],       0.05)
            w_sem   = st.slider("Semantic Match", 0.0, 1.0, WEIGHTS["semantic_match"],    0.05)
            w_exp   = st.slider("Experience",     0.0, 1.0, WEIGHTS["experience_match"],  0.05)
            w_ats   = st.slider("ATS Score",      0.0, 1.0, WEIGHTS["ats_score"],         0.05)
            w_edu   = st.slider("Education",      0.0, 1.0, WEIGHTS["education_match"],   0.05)
            total   = w_skill + w_sem + w_exp + w_ats + w_edu
            st.caption(f"Total: {total:.2f} {'✅' if abs(total-1.0)<0.01 else '⚠️ should sum to 1.0'}")
            WEIGHTS.update({"skill_match":w_skill,"semantic_match":w_sem,
                            "experience_match":w_exp,"ats_score":w_ats,"education_match":w_edu})

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("**🚨 Critical Skills** *(missing = ranking penalty)*")
        critical_raw = st.text_area(
            "Enter critical skills (one per line)",
            placeholder="python\nkubernetes\naws",
            height=100, label_visibility="collapsed",
        )
        critical_skills = [s.strip() for s in critical_raw.splitlines() if s.strip()]
        st.session_state["critical_skills"] = critical_skills
        if critical_skills:
            st.caption(f"🔴 {len(critical_skills)} critical skill(s) set")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("**🤖 Gemini Analysis Scope**")
        gemini_top_n = st.selectbox(
            "Run AI analysis for top N candidates",
            [3, 5, 10, 999], index=1,
            format_func=lambda x: f"Top {x}" if x != 999 else "All candidates",
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("**📋 How to Use**")
        st.caption("""
1. Upload Job Description (PDF/DOCX/TXT)
2. Set critical skills if required
3. Upload resumes (multiple OK)
4. Click **Analyze**
5. Use Gemini key for AI insights
6. Download Excel report
        """)

    # ── Header ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-title">AI Resume Matcher & ATS Screening Platform</div>
    <div class="hero-sub">Rank candidates intelligently with NLP, semantic AI, critical skill filtering, and Workday-style ATS scoring</div>
    """, unsafe_allow_html=True)

    # ── Upload Section ────────────────────────────────────────────────────────────
    col_jd, col_res = st.columns([1, 1], gap="large")
    with col_jd:
        st.markdown('<div class="section-header">📄 Job Description</div>', unsafe_allow_html=True)
        jd_file = st.file_uploader("Upload JD", type=["pdf","docx","txt"], label_visibility="collapsed")
    with col_res:
        st.markdown('<div class="section-header">📁 Resumes</div>', unsafe_allow_html=True)
        resume_files = st.file_uploader(
            "Upload Resumes", type=["pdf","docx","txt"],
            accept_multiple_files=True, label_visibility="collapsed",
        )

    # ── Filters ───────────────────────────────────────────────────────────────────
    with st.expander("🔍 Filter & Sort Options", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            min_match = st.slider("Min Match Score %", 0, 100, 0, 5)
        with fc2:
            min_exp   = st.slider("Min Experience (Years)", 0, 20, 0, 1)
        with fc3:
            sort_by   = st.selectbox("Sort By", ["Overall Score","ATS Score","Skill Match","Experience"])
        search_query = st.text_input("🔎 Search by name / email / skill / company", placeholder="e.g. python or john@email.com")

    analyze_btn = st.button("🚀 Analyze Resumes", type="primary", use_container_width=True)

    # ── Analysis ──────────────────────────────────────────────────────────────────
    if analyze_btn:
        if not jd_file:
            st.error("Please upload a Job Description.")
            return
        if not resume_files:
            st.error("Please upload at least one resume.")
            return

        with st.status("Parsing Job Description…", expanded=False) as status:
            try:
                jd_text = extract_text(jd_file)
                if not jd_text.strip():
                    st.error("Job Description is empty or unreadable.")
                    return
                jd_hash = hashlib.md5(jd_text.encode()).hexdigest()
                jd = parse_job_description(jd_text, jd_hash)
                status.update(label="✅ Job Description parsed", state="complete")
            except Exception as e:
                st.error(f"Failed to parse JD: {e}")
                return

        all_results: list[dict] = []
        errors: list[str] = []
        progress_bar = st.progress(0, text="Processing resumes…")
        n = len(resume_files)

        for idx, rf in enumerate(resume_files):
            progress_bar.progress(idx/n, text=f"Processing {rf.name}…")
            try:
                res_text = extract_text(rf)
                if not res_text.strip():
                    errors.append(f"{rf.name}: No readable text found.")
                    continue
                res_hash   = hashlib.md5(res_text.encode()).hexdigest()
                resume     = parse_resume(res_text, res_hash)
                crit_key   = hashlib.md5("|".join(sorted(critical_skills)).encode()).hexdigest()
                cache_key  = f"{res_hash}_{jd_hash}"
                match_result = match_resume_to_jd(resume, jd, cache_key, crit_key)
                all_results.append({
                    "resume": resume, "match_result": match_result,
                    "ai_analysis": {}, "filename": rf.name,
                })
            except ValueError as e:
                errors.append(f"{rf.name}: {e}")
            except Exception as e:
                errors.append(f"{rf.name}: Unexpected error — {str(e)[:120]}")

        progress_bar.progress(1.0, text="✅ Resume processing complete!")

        # Sort
        sort_map = {
            "Overall Score": lambda r: r["match_result"]["overall_score"],
            "ATS Score":     lambda r: r["match_result"]["ats_score"],
            "Skill Match":   lambda r: r["match_result"]["components"]["skill_match"],
            "Experience":    lambda r: r["resume"]["experience_years"],
        }
        all_results.sort(key=sort_map[sort_by], reverse=True)
        for i, r in enumerate(all_results):
            r["rank"] = i + 1

        # Gemini analysis for top N only
        if gemini_api_key:
            valid, _ = validate_gemini_key(gemini_api_key)
            if valid:
                top_n = all_results[:gemini_top_n]
                ai_progress = st.progress(0, text="Running Gemini AI analysis…")
                for idx, r in enumerate(top_n):
                    ai_progress.progress((idx+1)/max(len(top_n),1), text=f"Analyzing {r['resume']['name']}…")
                    r["ai_analysis"] = gemini_analyze(r["resume"], jd, r["match_result"], gemini_api_key)
                ai_progress.progress(1.0, text="✅ AI analysis complete!")

        if errors:
            with st.expander(f"⚠️ {len(errors)} file(s) had errors — click to view"):
                for err in errors:
                    st.caption(f"• {err}")

        if not all_results:
            st.error("No resumes could be processed. Please check your files and try again.")
            return

        st.session_state["all_results"] = all_results
        st.session_state["jd"]          = jd
        st.session_state["jd_hash"]     = jd_hash

    # ── Render Results ────────────────────────────────────────────────────────────
    if "all_results" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;color:#334155;">
          <div style="font-size:3rem;">📂</div>
          <div style="font-size:1.1rem;margin-top:12px;">Upload a Job Description and Resumes, then click Analyze</div>
        </div>
        """, unsafe_allow_html=True)
        return

    all_results = st.session_state["all_results"]
    jd          = st.session_state["jd"]
    jd_hash     = st.session_state.get("jd_hash", "")

    # Apply search
    if "search_query" not in dir():
        search_query = ""
    searched = candidate_search(all_results, search_query)

    # Apply numeric filters
    filtered = [
        r for r in searched
        if r["match_result"]["overall_score"]*100 >= min_match
        and r["resume"]["experience_years"] >= min_exp
    ]

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Dashboard KPIs ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Recruiter Dashboard</div>', unsafe_allow_html=True)

    scores   = [r["match_result"]["overall_score"]*100 for r in filtered]
    ats_vals = [r["match_result"]["ats_score"] for r in filtered]
    exps     = [r["resume"]["experience_years"] for r in filtered]
    strong   = sum(1 for s in scores if s >= 65)
    medium   = sum(1 for s in scores if 40 <= s < 65)
    weak     = sum(1 for s in scores if s < 40)

    render_kpi_row([
        ("Total Resumes",   len(all_results),  f"{len(filtered)} shown"),
        ("Strong Matches",  strong,            f"≥65% score"),
        ("Medium Matches",  medium,            "40–64% score"),
        ("Avg Match Score", f"{np.mean(scores):.1f}%" if scores else "—", None),
        ("Avg ATS Score",   f"{np.mean(ats_vals):.0f}" if ats_vals else "—", "out of 100"),
        ("Avg Experience",  f"{np.mean(exps):.1f} yrs" if exps else "—", None),
    ])

    # Bulk summary
    if filtered:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Bulk Recruiter Summary", expanded=False):
            top_candidate = filtered[0]
            all_missing = []
            for r in filtered:
                all_missing.extend(r["match_result"]["missing_skills"])
            top_missing = Counter(all_missing).most_common(8)

            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Strong Matches (≥65%)", strong)
                st.metric("Medium Matches (40–64%)", medium)
                st.metric("Weak Matches (<40%)", weak)
            with bc2:
                st.metric("Best Candidate", top_candidate["resume"]["name"])
                st.metric("Best Score", f"{top_candidate['match_result']['overall_score']*100:.1f}%")
                st.metric("Best ATS", f"{top_candidate['match_result']['ats_score']}/100")
            with bc3:
                st.markdown("**Top Missing Skills Across All Candidates**")
                for skill, count in top_missing:
                    st.caption(f"❌ {skill} — missing in {count} candidate(s)")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🏆 Rankings", "📈 Analytics", "🔬 Deep Analysis",
        "⚖️ Compare", "📋 JD Insights",
    ])

    # ── Tab 1: Rankings ───────────────────────────────────────────────────────────
    with tabs[0]:
        if not filtered:
            st.warning("No candidates match current filters.")
        else:
            st.markdown(f"**Showing {len(filtered)} candidate(s)**")
            if search_query:
                st.caption(f"🔎 Filtered by: *{search_query}*")

            # Top candidate banner
            top = filtered[0]
            top_badge, top_badge_cls = get_fit_badge(top["match_result"]["overall_score"]*100)
            st.success(
                f"🏆 **Top: {top['resume']['name']}** — "
                f"{top['match_result']['overall_score']*100:.1f}% match | "
                f"ATS {top['match_result']['ats_score']}/100 | "
                f"{top['resume']['experience_years']} yrs exp"
            )

            for r in filtered:
                res = r["resume"]
                mr  = r["match_result"]
                ai  = r.get("ai_analysis", {})
                score_pct = mr["overall_score"] * 100
                badge_label, badge_cls = get_fit_badge(score_pct)
                rank_icon = {1:"🥇",2:"🥈",3:"🥉"}.get(r["rank"], f"#{r['rank']}")
                crit_warn = f" | 🚨 {len(mr.get('critical_missing',[]))} critical missing" if mr.get("critical_missing") else ""

                with st.expander(
                    f"{rank_icon}  {res['name']}  —  {score_pct:.1f}% match  |  ATS {mr['ats_score']}/100  |  {res['experience_years']} yrs{crit_warn}",
                    expanded=(r["rank"] == 1),
                ):
                    # Badge row
                    st.markdown(
                        f'<span class="{badge_cls}">{badge_label}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("<br>", unsafe_allow_html=True)

                    c1, c2, c3 = st.columns([1.5, 1.5, 1])

                    with c1:
                        st.markdown("**📇 Contact**")
                        if res["email"]:    st.caption(f"📧 {res['email']}")
                        if res["phone"]:    st.caption(f"📱 {res['phone']}")
                        if res["linkedin"]: st.caption(f"🔗 {res['linkedin']}")
                        if res["github"]:   st.caption(f"💻 {res['github']}")
                        if res["location"]: st.caption(f"📍 {res['location']}")

                        st.markdown("**🏢 Experience**")
                        for comp in res["companies"][:4]:
                            st.caption(f"• {comp}")
                        for desig in res["designations"][:3]:
                            st.caption(f"  ↳ {desig}")
                        st.caption(f"**Total:** {res['experience_years']} years")

                        if res["education"]:
                            st.markdown("**🎓 Education**")
                            for edu in res["education"][:2]:
                                st.caption(f"• {edu[:120]}")
                        if res["certifications"]:
                            st.markdown("**📜 Certifications**")
                            for cert in res["certifications"][:2]:
                                st.caption(f"• {cert[:100]}")

                    with c2:
                        st.markdown("**📊 Scores**")
                        for label, val, help_txt in [
                            ("Overall Match", score_pct, "Weighted composite"),
                            ("Skill Match",   mr["components"]["skill_match"]*100,       "Weight 40%"),
                            ("Semantic",      mr["components"]["semantic_match"]*100,     "Weight 25%"),
                            ("Exp Match",     mr["components"]["experience_match"]*100,   "Weight 15%"),
                            ("Edu Match",     mr["components"]["education_match"]*100,    "Weight 10%"),
                        ]:
                            st.metric(label, f"{val:.1f}%", help=help_txt)
                            st.progress(val/100)
                        st.metric("ATS Score", f"{mr['ats_score']}/100")
                        st.progress(mr["ats_score"]/100)

                    with c3:
                        st.markdown("**🎯 Skills**")
                        render_skill_tags(mr["matched_skills"][:15],  "tag-match",   "✅ Matched")
                        render_skill_tags(mr["missing_skills"][:15],  "tag-missing", "❌ Missing")
                        if mr.get("critical_missing"):
                            render_skill_tags(mr["critical_missing"], "tag-critical", "🚨 Critical")
                        render_skill_tags(mr["additional_skills"][:8],"tag-extra",   "➕ Extra")

                    # ATS & Resume suggestions
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    sug_c1, sug_c2 = st.columns(2)
                    with sug_c1:
                        st.markdown("**🎯 ATS Improvement Tips**")
                        for tip in generate_ats_suggestions(res, jd, mr)[:4]:
                            st.markdown(f'<div class="ats-suggestion">💡 {tip}</div>', unsafe_allow_html=True)
                    with sug_c2:
                        st.markdown("**📝 Resume Improvement Tips**")
                        for tip in generate_resume_suggestions(res, jd, mr)[:4]:
                            st.markdown(f'<div class="resume-suggestion">✏️ {tip}</div>', unsafe_allow_html=True)

                    # AI Panel
                    if ai and "error" not in ai:
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        st.markdown('<div class="ai-panel">', unsafe_allow_html=True)
                        st.markdown("**🤖 Gemini AI Analysis**")
                        hire_color = {"Hire":"#34d399","Shortlist":"#fbbf24","Reject":"#f87171"}.get(
                            ai.get("hiring_decision",""), "#94a3b8"
                        )
                        st.markdown(
                            f'<span style="color:{hire_color};font-size:1.1rem;font-weight:700;">'
                            f'{ai.get("hiring_decision","")}</span> '
                            f'<span style="color:#94a3b8;font-size:0.85rem;">({ai.get("interview_recommendation","")})</span>',
                            unsafe_allow_html=True,
                        )
                        ai_c1, ai_c2 = st.columns(2)
                        with ai_c1:
                            if ai.get("candidate_summary"):
                                st.markdown('<div class="ai-section-title">Summary</div>', unsafe_allow_html=True)
                                st.caption(ai["candidate_summary"])
                            if ai.get("strengths"):
                                st.markdown('<div class="ai-section-title">Strengths</div>', unsafe_allow_html=True)
                                for s in ai["strengths"]:
                                    st.caption(f"✓ {s}")
                            if ai.get("positive_indicators"):
                                st.markdown('<div class="ai-section-title">Positive Indicators</div>', unsafe_allow_html=True)
                                for p in ai["positive_indicators"]:
                                    st.caption(f"★ {p}")
                        with ai_c2:
                            if ai.get("weaknesses"):
                                st.markdown('<div class="ai-section-title">Weaknesses</div>', unsafe_allow_html=True)
                                for w in ai["weaknesses"]:
                                    st.caption(f"⚠ {w}")
                            if ai.get("improvement_suggestions"):
                                st.markdown('<div class="ai-section-title">Improvement Suggestions</div>', unsafe_allow_html=True)
                                for s in ai["improvement_suggestions"]:
                                    st.caption(f"→ {s}")
                            if ai.get("red_flags"):
                                st.markdown('<div class="ai-section-title">Red Flags</div>', unsafe_allow_html=True)
                                for rf_item in ai["red_flags"]:
                                    st.caption(f"🚩 {rf_item}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    elif ai.get("error"):
                        st.caption(f"⚠️ AI analysis error: {ai['error']}")

                    with st.expander("📊 Score Breakdown Chart"):
                        st.plotly_chart(
                            chart_score_breakdown(mr),
                            use_container_width=True,
                            key=f"score_breakdown_{idx}"
                        )

    # ── Tab 2: Analytics ──────────────────────────────────────────────────────────
    with tabs[1]:
        if not filtered:
            st.warning("No candidates to visualize.")
        else:
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                st.plotly_chart(chart_match_scores(filtered), use_container_width=True)
            with r1c2:
                st.plotly_chart(chart_ats_scores(filtered), use_container_width=True)

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                st.plotly_chart(chart_experience_distribution(filtered), use_container_width=True)
            with r2c2:
                st.plotly_chart(chart_skill_frequency(filtered, jd), use_container_width=True)

            st.plotly_chart(chart_skill_heatmap(filtered, jd), use_container_width=True)

            if len(filtered) >= 2:
                st.plotly_chart(chart_radar(filtered[:5]), use_container_width=True)

            st.markdown('<div class="section-header">📋 Score Comparison Table</div>', unsafe_allow_html=True)
            table_data = []
            for r in filtered:
                mr = r["match_result"]
                ai = r.get("ai_analysis", {})
                badge_label, _ = get_fit_badge(mr["overall_score"]*100)
                table_data.append({
                    "Rank": r["rank"],
                    "Name": r["resume"]["name"],
                    "Fit": badge_label,
                    "Overall %": round(mr["overall_score"]*100, 1),
                    "Skill %": round(mr["components"]["skill_match"]*100, 1),
                    "Semantic %": round(mr["components"]["semantic_match"]*100, 1),
                    "Exp %": round(mr["components"]["experience_match"]*100, 1),
                    "ATS": mr["ats_score"],
                    "Edu %": round(mr["components"]["education_match"]*100, 1),
                    "Exp (Yrs)": r["resume"]["experience_years"],
                    "Critical Missing": len(mr.get("critical_missing", [])),
                    "Decision": ai.get("hiring_decision", "—"),
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            xlsx_bytes = build_excel_report(all_results, jd)
            st.download_button(
                "📥 Download Full Excel Report (6 Sheets)",
                data=xlsx_bytes,
                file_name=f"ats_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary",
            )

    # ── Tab 3: Deep Analysis ──────────────────────────────────────────────────────
    with tabs[2]:
        if not filtered:
            st.warning("No candidates to analyze.")
        else:
            selected_name = st.selectbox(
                "Select Candidate", [r["resume"]["name"] for r in filtered]
            )
            r = next(x for x in filtered if x["resume"]["name"] == selected_name)
            res, mr, ai = r["resume"], r["match_result"], r.get("ai_analysis", {})

            badge_label, badge_cls = get_fit_badge(mr["overall_score"]*100)
            st.markdown(f'<span class="{badge_cls}">{badge_label}</span><br><br>', unsafe_allow_html=True)

            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.markdown("**🔬 Resume Details**")
                for k, v in {
                    "Name": res["name"], "Email": res["email"], "Phone": res["phone"],
                    "LinkedIn": res["linkedin"], "GitHub": res["github"],
                    "Location": res["location"],
                    "Experience": f"{res['experience_years']} years",
                }.items():
                    if v:
                        st.text_input(k, v, disabled=True)

                with st.expander("🏢 Companies & Roles"):
                    for c in res["companies"]:
                        st.caption(f"• {c}")
                    for d in res["designations"]:
                        st.caption(f"  ↳ {d}")
                with st.expander("🎓 Education"):
                    for e in res["education"]:
                        st.caption(f"• {e}")
                with st.expander("📜 Certifications"):
                    for c in res["certifications"]:
                        st.caption(f"• {c}")
                with st.expander("🚀 Projects"):
                    for p in res["projects"]:
                        st.caption(f"• {p}")

                st.markdown("**🎯 ATS Improvement Suggestions**")
                for tip in generate_ats_suggestions(res, jd, mr):
                    st.markdown(f'<div class="ats-suggestion">💡 {tip}</div>', unsafe_allow_html=True)

                st.markdown("**📝 Resume Improvement Suggestions**")
                for tip in generate_resume_suggestions(res, jd, mr):
                    st.markdown(f'<div class="resume-suggestion">✏️ {tip}</div>', unsafe_allow_html=True)

                if mr.get("critical_missing"):
                    st.markdown("**🚨 Critical Skills Missing**")
                    for skill in mr["critical_missing"]:
                        st.markdown(f'<div class="critical-missing">🚨 {skill} — required by JD and marked as critical</div>', unsafe_allow_html=True)

            with dcol2:
                st.plotly_chart(
                    chart_score_breakdown(mr),
                    use_container_width=True,
                    key=f"score_breakdown_{idx}"
                )

                st.markdown("**🎯 Skill Gap Analysis**")
                skill_data_rows = (
                    [{"Skill": s, "Status": "Matched"} for s in mr["matched_skills"]] +
                    [{"Skill": s, "Status": "Missing"} for s in mr["missing_skills"][:10]]
                )
                if skill_data_rows:
                    skill_df = pd.DataFrame(skill_data_rows)
                    fig_skills = px.bar(
                        skill_df, x="Skill", color="Status",
                        color_discrete_map={"Matched":"#34d399","Missing":"#f87171"},
                        title="Skill Status",
                    )
                    fig_skills.update_layout(
                        paper_bgcolor=PAPER_BG, plot_bgcolor=CHART_BG,
                        font_color=FONT_COLOR, margin=dict(l=10,r=10,t=40,b=80),
                        xaxis_tickangle=-45,
                    )
                    st.plotly_chart(fig_skills, use_container_width=True)

            if ai and "error" not in ai:
                st.markdown('<div class="section-header">🤖 Full Gemini AI Report</div>', unsafe_allow_html=True)
                st.json(ai)

    # ── Tab 4: Compare ────────────────────────────────────────────────────────────
    with tabs[3]:
        if len(filtered) < 2:
            st.info("Upload at least 2 resumes to compare candidates side by side.")
        else:
            candidate_names = [r["resume"]["name"] for r in filtered]
            cmp_col1, cmp_col2 = st.columns(2)
            with cmp_col1:
                cand_a = st.selectbox("Candidate A", candidate_names, index=0, key="cmp_a")
            with cmp_col2:
                cand_b = st.selectbox("Candidate B", candidate_names,
                                      index=min(1, len(candidate_names)-1), key="cmp_b")

            ra = next(x for x in filtered if x["resume"]["name"] == cand_a)
            rb = next(x for x in filtered if x["resume"]["name"] == cand_b)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### ⚖️ Side-by-Side Comparison")

            compare_metrics = [
                ("Overall Match %",   f"{ra['match_result']['overall_score']*100:.1f}%",  f"{rb['match_result']['overall_score']*100:.1f}%"),
                ("Skill Match %",     f"{ra['match_result']['components']['skill_match']*100:.1f}%", f"{rb['match_result']['components']['skill_match']*100:.1f}%"),
                ("Semantic Match %",  f"{ra['match_result']['components']['semantic_match']*100:.1f}%", f"{rb['match_result']['components']['semantic_match']*100:.1f}%"),
                ("Exp Match %",       f"{ra['match_result']['components']['experience_match']*100:.1f}%", f"{rb['match_result']['components']['experience_match']*100:.1f}%"),
                ("ATS Score",         f"{ra['match_result']['ats_score']}/100", f"{rb['match_result']['ats_score']}/100"),
                ("Education Match %", f"{ra['match_result']['components']['education_match']*100:.1f}%", f"{rb['match_result']['components']['education_match']*100:.1f}%"),
                ("Experience (Yrs)",  f"{ra['resume']['experience_years']}", f"{rb['resume']['experience_years']}"),
                ("Matched Skills",    f"{len(ra['match_result']['matched_skills'])}", f"{len(rb['match_result']['matched_skills'])}"),
                ("Missing Skills",    f"{len(ra['match_result']['missing_skills'])}", f"{len(rb['match_result']['missing_skills'])}"),
                ("Critical Missing",  f"{len(ra['match_result'].get('critical_missing',[]))}", f"{len(rb['match_result'].get('critical_missing',[]))}"),
                ("Email",             "✅" if ra["resume"]["email"] else "❌", "✅" if rb["resume"]["email"] else "❌"),
                ("LinkedIn",          "✅" if ra["resume"]["linkedin"] else "❌", "✅" if rb["resume"]["linkedin"] else "❌"),
                ("GitHub",            "✅" if ra["resume"]["github"] else "❌", "✅" if rb["resume"]["github"] else "❌"),
                ("Fit Badge",         get_fit_badge(ra["match_result"]["overall_score"]*100)[0],
                                      get_fit_badge(rb["match_result"]["overall_score"]*100)[0]),
            ]

            cmp_df = pd.DataFrame(compare_metrics, columns=["Metric", cand_a, cand_b])
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            # Radar overlay
            st.plotly_chart(chart_radar([ra, rb]), use_container_width=True)

            # Unique skills each has that the other doesn't
            skills_a = set(s.lower() for s in ra["resume"]["skills"])
            skills_b = set(s.lower() for s in rb["resume"]["skills"])
            unique_a = list(skills_a - skills_b)[:10]
            unique_b = list(skills_b - skills_a)[:10]
            uc1, uc2 = st.columns(2)
            with uc1:
                st.markdown(f"**Skills unique to {cand_a}**")
                render_skill_tags(unique_a, "tag-extra", "Unique")
            with uc2:
                st.markdown(f"**Skills unique to {cand_b}**")
                render_skill_tags(unique_b, "tag-extra", "Unique")

    # ── Tab 5: JD Insights ────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown(f"**Role:** {jd['role']}")
        st.metric("Required Experience", f"{jd['required_experience_years']} years")

        jc1, jc2 = st.columns(2)
        with jc1:
            with st.expander("🎯 Required Skills", expanded=True):
                for s in jd["required_skills"][:30]:
                    is_crit = any(s.lower() == c.lower() for c in critical_skills)
                    prefix  = "🚨 " if is_crit else "• "
                    st.caption(f"{prefix}{s}")
            with st.expander("🎓 Required Education"):
                for e in jd["required_education"]:
                    st.caption(f"• {e}")
        with jc2:
            with st.expander("📋 Responsibilities"):
                st.caption(jd["responsibilities"] or "Not extracted")
            with st.expander("✅ Requirements"):
                st.caption(jd["requirements"] or "Not extracted")
            with st.expander("⭐ Nice to Have"):
                st.caption(jd["nice_to_have"] or "Not extracted")

        if all_results:
            st.markdown('<div class="section-header">📊 Skill Coverage Across Candidates</div>', unsafe_allow_html=True)
            skill_coverage = {}
            for jd_skill in jd["required_skills"][:20]:
                skill_coverage[jd_skill] = sum(
                    1 for r in all_results
                    if jd_skill in r["match_result"]["matched_skills"]
                )
            if skill_coverage:
                cov_df = pd.DataFrame(
                    list(skill_coverage.items()), columns=["Skill","Candidates with Skill"]
                ).sort_values("Candidates with Skill", ascending=False)
                fig_cov = px.bar(
                    cov_df, x="Candidates with Skill", y="Skill", orientation="h",
                    color="Candidates with Skill",
                    color_continuous_scale=[[0,"#f87171"],[0.5,"#fbbf24"],[1,"#34d399"]],
                    title=f"Required Skill Coverage (of {len(all_results)} candidates)",
                )
                fig_cov.update_layout(
                    paper_bgcolor=PAPER_BG, plot_bgcolor=CHART_BG, font_color=FONT_COLOR,
                    coloraxis_showscale=False, margin=dict(l=10,r=10,t=40,b=10),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_cov, use_container_width=True)

            st.plotly_chart(chart_skill_heatmap(all_results[:15], jd), use_container_width=True)


if __name__ == "__main__":
    main()
