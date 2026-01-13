"""Resume Analyzer & Tailor - LLM-powered resume analysis and optimization."""

import streamlit as st
import fitz  # PyMuPDF
import docx
import re
import json
import hashlib
import logging
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

@dataclass
class AppConfig:
    """Application configuration constants."""
    PAGE_TITLE = "Resume Analyzer & Tailor"
    APP_TITLE = "🧭 LLM-Powered Resume Analyzer & Tailor"
    LAYOUT = "wide"
    
    # OpenAI settings
    MODEL = "gpt-4o-mini"
    ANALYSIS_TEMP = 0.12
    TAILOR_TEMP = 0.28
    INTERVIEW_TEMP = 0.25
    MAX_TOKENS_ANALYSIS = 1200
    MAX_TOKENS_TAILOR = 1200
    MAX_TOKENS_ATS = 700
    MAX_TOKENS_INTERVIEW = 1000
    
    # Interview prep
    BASE_QUESTIONS = 6
    MAX_QUESTIONS = 10
    
    # Session state keys
    RESUME_TEXT = "resume_text"
    JOB_DESCRIPTION = "job_description"
    ANALYSIS_RAW = "analysis_raw"
    ANALYSIS_JSON = "analysis_json"
    MATCH_SCORE = "match_score"
    TAILORED_RESUME = "tailored_resume"
    ATS_REPORT = "ats_report"
    INTERVIEW_QUESTIONS = "interview_questions"
    LAST_FILENAME = "last_filename"
    PAGE = "page"
    
    # Pages
    PAGE_HOME = "Home"
    PAGE_ANALYZE = "Analyze Resume"
    PAGE_TAILOR = "Tailor Resume"
    PAGE_ATS = "ATS Check"
    PAGE_INTERVIEW = "Interview Prep"

# Color thresholds
SCORE_THRESHOLDS = {
    "green": 70,
    "orange": 50,
    "red": 0,
}

ATS_THRESHOLDS = {
    "green": 80,
    "orange": 60,
    "red": 0,
}

# Session state defaults
SESSION_DEFAULTS = {
    AppConfig.RESUME_TEXT: None,
    AppConfig.JOB_DESCRIPTION: None,
    AppConfig.ANALYSIS_RAW: None,
    AppConfig.ANALYSIS_JSON: None,
    AppConfig.MATCH_SCORE: None,
    AppConfig.TAILORED_RESUME: None,
    AppConfig.ATS_REPORT: None,
    AppConfig.INTERVIEW_QUESTIONS: None,
    AppConfig.LAST_FILENAME: None,
    AppConfig.PAGE: AppConfig.PAGE_HOME,
}

# Cache settings
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
ATS_CACHE_FILE = CACHE_DIR / "ats_cache.json"

st.set_page_config(page_title=AppConfig.PAGE_TITLE, layout=AppConfig.LAYOUT)
st.title(AppConfig.APP_TITLE)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_text(file) -> str:
    """Extract text from PDF or DOCX file.
    
    Args:
        file: Uploaded file object (PDF or DOCX).
        
    Returns:
        Extracted text as string.
    """
    name = file.name.lower()
    try:
        if name.endswith(".pdf"):
            pdf = fitz.open(stream=file.read(), filetype="pdf")
            return "".join(page.get_text() for page in pdf)
        elif name.endswith(".docx"):
            d = docx.Document(file)
            return "\n".join(p.text for p in d.paragraphs)
    except Exception as e:
        logger.error(f"Error extracting text from {name}: {e}")
    return ""

def clamp_int(n: Any, lo: int = 0, hi: int = 100) -> Optional[int]:
    """Clamp an integer value between lo and hi bounds.
    
    Args:
        n: Value to clamp.
        lo: Lower bound (default 0).
        hi: Upper bound (default 100).
        
    Returns:
        Clamped integer or None if conversion fails.
    """
    try:
        return max(lo, min(hi, int(n)))
    except (ValueError, TypeError):
        return None

def get_score_color(score: Optional[int], thresholds: Dict[str, int]) -> str:
    """Get color for a score based on thresholds.
    
    Args:
        score: Score value to evaluate.
        thresholds: Dict with keys 'green', 'orange', 'red' and threshold values.
        
    Returns:
        Color name as string.
    """
    if score is None:
        return "gray"
    if score >= thresholds.get("green", 70):
        return "green"
    if score >= thresholds.get("orange", 50):
        return "orange"
    return "red"

def resume_hash(text: str) -> str:
    """Generate SHA256 hash of resume text.
    
    Args:
        text: Resume text to hash.
        
    Returns:
        Hex digest of SHA256 hash.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def try_parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from text, handling various formats.
    
    Args:
        text: Text potentially containing JSON.
        
    Returns:
        Parsed dict or None if parsing fails.
    """
    if not text:
        return None
    
    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON from wrapped text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    
    return None

# ============================================================================
# UI RENDERING FUNCTIONS
# ============================================================================

def render_score_display(col, label: str, score: int, show_bar: bool = True) -> None:
    """Render a score display with optional progress bar.
    
    Args:
        col: Streamlit column to render in.
        label: Label for the score.
        score: Score value (0-100).
        show_bar: Whether to show progress bar.
    """
    color = get_score_color(score, SCORE_THRESHOLDS)
    col.markdown(
        f"<div style='font-size:40px; font-weight:800; color:{color};'>{label}: {score}%</div>",
        unsafe_allow_html=True,
    )
    if show_bar:
        col.markdown(
            f"""
            <div style="background-color:#e6e6e6; border-radius:10px; height:20px; width:100%; margin:6px 0 12px 0;">
                <div style="background-color:{color}; width:{score}%; height:100%; border-radius:10px;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_ats_score(col, label: str, score: Optional[int]) -> None:
    """Render ATS score display.
    
    Args:
        col: Streamlit column to render in.
        label: Label for the score.
        score: Score value (0-100) or None.
    """
    if score is None:
        color = "gray"
    else:
        color = get_score_color(score, ATS_THRESHOLDS)
    col.markdown(
        f"<div style='font-size:40px; font-weight:800; color:{color};'>{label}: {score}%</div>",
        unsafe_allow_html=True,
    )

# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def _load_ats_cache() -> Dict[str, Any]:
    """Load ATS cache from file."""
    if ATS_CACHE_FILE.exists():
        try:
            return json.loads(ATS_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load ATS cache: {e}")
            return {}
    return {}

def _save_ats_cache(obj: Dict[str, Any]) -> None:
    """Save ATS cache to file."""
    try:
        ATS_CACHE_FILE.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to save ATS cache: {e}")

def ats_get_cached(h: str) -> Optional[Dict[str, Any]]:
    """Get cached ATS result by hash."""
    obj = _load_ats_cache()
    return obj.get(h)

def ats_save_cached(h: str, data: Dict[str, Any]) -> None:
    """Save ATS result to cache."""
    obj = _load_ats_cache()
    obj[h] = {**data, "cached_at": datetime.utcnow().isoformat()}
    _save_ats_cache(obj)

# ============================================================================
# OPENAI API FUNCTIONS
# ============================================================================

def _call_openai(prompt: str, temperature: float, max_tokens: int) -> str:
    """Generic function to call OpenAI API.
    
    Args:
        prompt: The prompt to send to OpenAI.
        temperature: Temperature parameter for the API.
        max_tokens: Maximum tokens for the response.
        
    Returns:
        Response content as string.
        
    Raises:
        Exception: If API call fails.
    """
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=AppConfig.MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise

def run_structured_analysis() -> str:
    """Run structured resume analysis against job description."""
    prompt = f"""
You are a resume analysis engine. Compare the RESUME and JOB_DESCRIPTION and output STRICT JSON only with this schema:

{{
  "match_score": <integer 0-100>,
  "sections": {{
    "summary": {{"score": <0-100>, "comment": "<short comment>"}},
    "skills": {{"score": <0-100>, "comment": "<short comment>"}},
    "experience": {{"score": <0-100>, "comment": "<short comment>"}},
    "education": {{"score": <0-100>, "comment": "<short comment>"}}
  }},
  "matches": ["<keywords present in resume>"],
  "missing_keywords": [{{"keyword":"<kw>", "reason":"<why relevant>"}}],
  "recommended_tools": [{{"tool":"<tool>", "reason":"<why add it>"}}],
  "recommendations": ["<actionable points prioritized>"],
  "quick_summary": "<one-line summary>",
  "full_assessment": "<detailed multi-paragraph assessment>"
}}

Rules:
- Output JSON only. No extra surrounding commentary.
- Use numeric integer scores.
- If a section is not present, set its score to 0 and an empty comment.

RESUME:
{st.session_state.get(AppConfig.RESUME_TEXT, "")}

JOB_DESCRIPTION:
{st.session_state.get(AppConfig.JOB_DESCRIPTION, "")}
"""
    return _call_openai(prompt, AppConfig.ANALYSIS_TEMP, AppConfig.MAX_TOKENS_ANALYSIS)

def call_openai_tailor() -> str:
    """Generate a tailored resume draft."""
    prompt = f"""
You are an expert resume writer. Using the RESUME and JOB_DESCRIPTION below, produce a tailored resume draft that:
- keeps factual content from RESUME (do not invent new jobs or dates),
- emphasizes experiences and keywords relevant to JOB_DESCRIPTION,
- outputs ONLY the revised resume text (no extra commentary).

RESUME:
{st.session_state.get(AppConfig.RESUME_TEXT, "")}

JOB_DESCRIPTION:
{st.session_state.get(AppConfig.JOB_DESCRIPTION, "")}
"""
    return _call_openai(prompt, AppConfig.TAILOR_TEMP, AppConfig.MAX_TOKENS_TAILOR)

def call_openai_ats(resume_text: str) -> Dict[str, Any]:
    """Evaluate ATS friendliness of resume."""
    prompt = f"""
You are an expert on resume parsing and Applicant Tracking Systems (ATS). Given the RESUME (plain text), return STRICT JSON with:
{{
  "overall_score": <integer 0-100>,
  "issues": [
    {{"problem":"<short>", "recommendation":"<short>"}}
  ],
  "summary":"<short summary>"
}}

Guidance:
- Consider common ATS pitfalls: images, tables, multi-column layouts, special characters, long URLs, uncommon fonts, missing headings.
- Provide concise remediation for each issue.

RESUME:
{resume_text}
"""
    resp = _call_openai(prompt, AppConfig.ANALYSIS_TEMP, AppConfig.MAX_TOKENS_ATS)
    parsed = try_parse_json_from_text(resp)
    return parsed if parsed else {"overall_score": None, "issues": [], "summary": resp}

def call_openai_interview(num_questions: int) -> Dict[str, Any]:
    """Generate interview questions based on resume and job description."""
    prompt = f"""
You are an interviewer and career coach. Using the JOB_DESCRIPTION and the RESUME below:
- Generate {num_questions} role-specific interview questions.
- For each question include a short suggested bullet-answer derived from the RESUME (do NOT invent facts).
- Output JSON:
{{ "questions": [ {{"q":"<question>", "suggested_answer":"<short answer>"}} ] }}

JOB_DESCRIPTION:
{st.session_state.get(AppConfig.JOB_DESCRIPTION, "")}

RESUME:
{st.session_state.get(AppConfig.RESUME_TEXT, "")}
"""
    resp = _call_openai(prompt, AppConfig.INTERVIEW_TEMP, AppConfig.MAX_TOKENS_INTERVIEW)
    parsed = try_parse_json_from_text(resp)
    return parsed if parsed else {"questions": []}

# ============================================================================
# REPORT GENERATION
# ============================================================================

def _format_keywords(keywords: List[Any]) -> str:
    """Format keywords list handling both simple and dict formats."""
    if not keywords:
        return "N/A"
    if isinstance(keywords[0], dict):
        return "\n".join([f"{k.get('keyword', k.get('tool'))}: {k.get('reason', '')}" for k in keywords])
    return ", ".join(keywords)

def generate_report_bytes() -> bytes:
    """Generate a PDF or text report of the analysis."""
    title = "Resume Analysis Report"
    sections = {}
    aj = st.session_state.get(AppConfig.ANALYSIS_JSON)
    ms = st.session_state.get(AppConfig.MATCH_SCORE)
    
    sections["Overall Match Score"] = f"{ms}%" if ms is not None else "N/A"
    
    if aj and isinstance(aj, dict):
        sec_lines = []
        for sname, sdata in aj.get("sections", {}).items():
            score = sdata.get("score", "N/A")
            comment = sdata.get("comment", "")
            sec_lines.append(f"{sname.title()}: {score}%\n{comment}")
        sections["Section-wise Breakdown"] = "\n\n".join(sec_lines)
        sections["Matches"] = "\n".join(aj.get("matches", [])) if aj.get("matches") else "N/A"
        sections["Missing Keywords"] = _format_keywords(aj.get("missing_keywords", []))
        sections["Recommended Tools"] = _format_keywords(aj.get("recommended_tools", []))
        recs = aj.get("recommendations", [])
        sections["Recommendations"] = "\n".join(recs) if recs else "N/A"
        sections["Full Assessment"] = aj.get("full_assessment", "") or st.session_state.get(AppConfig.ANALYSIS_RAW, "")
    else:
        sections["Analysis Raw"] = st.session_state.get(AppConfig.ANALYSIS_RAW, "")

    if st.session_state.get(AppConfig.TAILORED_RESUME):
        sections["Tailored Resume"] = st.session_state.get(AppConfig.TAILORED_RESUME)

    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=16, style='B')
        pdf.cell(0, 8, title, ln=True)
        pdf.ln(4)
        pdf.set_font("Arial", size=11)
        for heading, content in sections.items():
            pdf.set_font("Arial", size=12, style='B')
            pdf.multi_cell(0, 7, heading)
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 6, str(content))
            pdf.ln(3)
        return pdf.output(dest="S").encode("latin-1")
    except Exception as e:
        logger.warning(f"PDF generation failed, falling back to text: {e}")
        text = [title, "\n"]
        for heading, content in sections.items():
            text.append(heading)
            text.append(str(content))
            text.append("\n\n")
        return "\n".join(text).encode("utf-8")

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def initialize_session_state() -> None:
    """Initialize session state with default values."""
    for key, default_value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# ============================================================================
# NAVIGATION & SIDEBAR
# ============================================================================

def render_sidebar() -> str:
    """Render sidebar navigation and return selected page."""
    st.sidebar.title("Navigation")
    page = st.session_state.get(AppConfig.PAGE, AppConfig.PAGE_HOME)
    
    if st.sidebar.button("🏠 Home"):
        page = AppConfig.PAGE_HOME
    if st.sidebar.button("📊 Analyze Resume"):
        page = AppConfig.PAGE_ANALYZE
    if st.sidebar.button("✍️ Tailor Resume"):
        page = AppConfig.PAGE_TAILOR
    if st.sidebar.button("✅ ATS Check"):
        page = AppConfig.PAGE_ATS
    if st.sidebar.button("🎤 Interview Prep"):
        page = AppConfig.PAGE_INTERVIEW

    st.sidebar.markdown("<div style='height:220px'></div>", unsafe_allow_html=True)

    if st.sidebar.button("🔄 Restart Session"):
        keys_to_clear = [
            AppConfig.RESUME_TEXT,
            AppConfig.JOB_DESCRIPTION,
            AppConfig.ANALYSIS_RAW,
            AppConfig.ANALYSIS_JSON,
            AppConfig.MATCH_SCORE,
            AppConfig.TAILORED_RESUME,
            AppConfig.ATS_REPORT,
            AppConfig.INTERVIEW_QUESTIONS,
            AppConfig.LAST_FILENAME,
        ]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        page = AppConfig.PAGE_HOME
        try:
            st.rerun()
        except Exception:
            st.markdown("<script>window.location.reload()</script>", unsafe_allow_html=True)
    
    st.session_state[AppConfig.PAGE] = page
    return page

# ============================================================================
# PAGE RENDERING FUNCTIONS
# ============================================================================

def render_home() -> None:
    """Render the home page."""
    st.header("Welcome — Resume Analyzer & Tailor")
    st.markdown(
        """
        **What you can do:**  
        1. Analyze Resume — upload resume (PDF/DOCX) and paste JD to get an overall match score and section breakdown.  
        2. Tailor Resume — generate a tailored draft based on the analysis (keeps facts).  
        3. ATS Check — evaluate ATS-friendliness (resume-only).  
        4. Interview Prep — generate tailored interview questions & suggested answers.
        """
    )
    st.info("Start with **Analyze Resume** in the sidebar. Only that page accepts uploads and JD input.")

def reset_analysis_state() -> None:
    """Reset analysis-dependent session state when a new file is uploaded."""
    st.session_state[AppConfig.ANALYSIS_RAW] = None
    st.session_state[AppConfig.ANALYSIS_JSON] = None
    st.session_state[AppConfig.MATCH_SCORE] = None
    st.session_state[AppConfig.TAILORED_RESUME] = None
    st.session_state[AppConfig.ATS_REPORT] = None
    st.session_state[AppConfig.INTERVIEW_QUESTIONS] = None

def render_analyze_resume() -> None:
    """Render the analyze resume page."""
    st.header("📊 Analyze Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    job_description = st.text_area("Paste the Job Description here", height=220, placeholder="Paste full JD text here...")

    if uploaded_file is not None:
        if st.session_state.get(AppConfig.LAST_FILENAME) != uploaded_file.name:
            st.session_state[AppConfig.RESUME_TEXT] = extract_text(uploaded_file)
            st.session_state[AppConfig.LAST_FILENAME] = uploaded_file.name
            reset_analysis_state()

    if job_description and job_description.strip():
        st.session_state[AppConfig.JOB_DESCRIPTION] = job_description

    if st.button("Analyze Resume"):
        if not st.session_state.get(AppConfig.RESUME_TEXT) or not st.session_state.get(AppConfig.JOB_DESCRIPTION):
            st.error("Please upload a resume and paste the job description first.")
        else:
            try:
                with st.spinner("Analyzing (structured)..."):
                    raw = run_structured_analysis()
                st.session_state[AppConfig.ANALYSIS_RAW] = raw
                parsed = try_parse_json_from_text(raw)
                st.session_state[AppConfig.ANALYSIS_JSON] = parsed
                if parsed and isinstance(parsed, dict):
                    st.session_state[AppConfig.MATCH_SCORE] = clamp_int(parsed.get("match_score"))
                else:
                    m = re.search(r"(\d{1,3})\s*%", raw)
                    st.session_state[AppConfig.MATCH_SCORE] = clamp_int(m.group(1)) if m else None
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                logger.error(f"Analysis error: {e}")

    if st.session_state.get(AppConfig.ANALYSIS_JSON):
        render_analysis_results()
    elif st.session_state.get(AppConfig.ANALYSIS_RAW):
        st.info("Analysis in progress or raw output available:")
        st.write(st.session_state.get(AppConfig.ANALYSIS_RAW))

def render_analysis_results() -> None:
    """Render structured analysis results."""
    aj = st.session_state[AppConfig.ANALYSIS_JSON]
    score = st.session_state.get(AppConfig.MATCH_SCORE)

    col_left, col_right = st.columns([1, 1])
    if score is not None:
        render_score_display(col_left, "Match Score", score)
    if aj.get("quick_summary"):
        col_right.markdown("**Quick summary**")
        col_right.write(aj.get("quick_summary"))

    st.subheader("Section-wise breakdown")
    sections = aj.get("sections", {})
    ordered = [("summary", "Summary"), ("skills", "Skills"), ("experience", "Experience"), ("education", "Education")]
    for i in range(0, len(ordered), 2):
        a_key, a_label = ordered[i]
        b_key, b_label = ordered[i + 1]
        col_a, col_b = st.columns(2)
        a_data = sections.get(a_key, {}) if sections else {}
        b_data = sections.get(b_key, {}) if sections else {}
        a_score = clamp_int(a_data.get("score")) if a_data else None
        b_score = clamp_int(b_data.get("score")) if b_data else None
        
        _render_section_score(col_a, a_label, a_score, a_data)
        _render_section_score(col_b, b_label, b_score, b_data)

    st.subheader("Detailed analysis & suggestions")
    if aj.get("matches"):
        st.markdown("**What in your resume already matches the JD:**")
        st.write(", ".join(aj.get("matches")))
    
    if aj.get("missing_keywords"):
        st.markdown("**Missing keywords / skills (consider adding these):**")
        missing = aj.get("missing_keywords")
        _render_keyword_list(missing)
    
    if aj.get("recommended_tools"):
        st.markdown("**Recommended tools / technologies to highlight:**")
        tools = aj.get("recommended_tools")
        _render_keyword_list(tools, key_name="tool")
    
    if aj.get("recommendations"):
        st.markdown("**Actionable recommendations (prioritized):**")
        for r in aj.get("recommendations"):
            st.write(f"- {r}")

    st.markdown("---")
    st.markdown("**Full assessment**")
    if aj.get("full_assessment"):
        st.write(aj.get("full_assessment"))
    else:
        st.write(st.session_state.get(AppConfig.ANALYSIS_RAW, ""))

def _render_section_score(col, label: str, score: Optional[int], data: Dict[str, Any]) -> None:
    """Render a section score with progress bar and comment."""
    if score is not None:
        col.write(f"**{label}** — {score}%")
        col.progress(score / 100.0)
        if data.get("comment"):
            col.caption(data.get("comment"))
    else:
        col.write(f"**{label}** — N/A")

def _render_keyword_list(keywords: List[Any], key_name: str = "keyword") -> None:
    """Render a list of keywords or tools."""
    if isinstance(keywords, list) and keywords and isinstance(keywords[0], dict):
        for k in keywords:
            st.write(f"- **{k.get(key_name)}** — {k.get('reason', '')}")
    else:
        st.write(", ".join(str(k) for k in keywords))

def render_ats_check() -> None:
    """Render the ATS check page."""
    st.header("✅ ATS Friendliness Check")
    st.write("Evaluates ATS compatibility for the uploaded resume (uses stored resume).")
    
    if not st.session_state.get(AppConfig.RESUME_TEXT):
        st.info("Upload a resume first on the Analyze Resume page to run ATS checks.")
    else:
        if st.button("Run ATS Check"):
            try:
                text = st.session_state.get(AppConfig.RESUME_TEXT)
                h = resume_hash(text)
                cached = ats_get_cached(h)
                if cached:
                    st.success("Using cached ATS result (resume unchanged).")
                    result = cached
                else:
                    with st.spinner("Running ATS check..."):
                        result = call_openai_ats(text)
                    ats_save_cached(h, result)
                st.session_state[AppConfig.ATS_REPORT] = result
            except Exception as e:
                st.error(f"ATS check failed: {e}")
                logger.error(f"ATS check error: {e}")

        if st.session_state.get(AppConfig.ATS_REPORT):
            ar = st.session_state[AppConfig.ATS_REPORT]
            overall = clamp_int(ar.get("overall_score")) if isinstance(ar, dict) else None
            if overall is not None:
                render_ats_score(st, "ATS Score", overall)
            
            st.subheader("Issues & Recommendations")
            if isinstance(ar, dict):
                for it in ar.get("issues", []):
                    st.write(f"- **{it.get('problem', 'Issue')}** — {it.get('recommendation', '')}")
                st.markdown("---")
                st.write(ar.get("summary", ""))
            else:
                st.write(ar)

def render_interview_prep() -> None:
    """Render the interview prep page."""
    st.header("🎤 Interview Prep")
    st.write("Generate tailored interview questions & suggested answers (uses stored resume + JD).")
    
    if not st.session_state.get(AppConfig.RESUME_TEXT) or not st.session_state.get(AppConfig.JOB_DESCRIPTION):
        st.info("Upload a resume and paste a job description on the Analyze Resume page first.")
    else:
        # Determine number of questions from JD length and match score
        jd = st.session_state.get(AppConfig.JOB_DESCRIPTION, "")
        jd_len = len(jd.split())
        match = st.session_state.get(AppConfig.MATCH_SCORE) or 0
        
        extra = 0
        if jd_len > 600:
            extra += 2
        elif jd_len > 300:
            extra += 1
        if match >= 80:
            extra += 2
        elif match >= 60:
            extra += 1
        num_questions = min(AppConfig.MAX_QUESTIONS, AppConfig.BASE_QUESTIONS + extra)

        if st.button("Generate Interview Questions"):
            try:
                with st.spinner("Generating interview questions..."):
                    q_json = call_openai_interview(num_questions)
                st.session_state[AppConfig.INTERVIEW_QUESTIONS] = q_json
            except Exception as e:
                st.error(f"Interview generation failed: {e}")
                logger.error(f"Interview generation error: {e}")

        if st.session_state.get(AppConfig.INTERVIEW_QUESTIONS):
            iq = st.session_state[AppConfig.INTERVIEW_QUESTIONS]
            if isinstance(iq, dict) and iq.get("questions"):
                for idx, q in enumerate(iq.get("questions", []), start=1):
                    st.markdown(f"**Q{idx}: {q.get('q')}**")
                    st.markdown(f"- Suggested answer: {q.get('suggested_answer')}")
                    st.markdown("---")
            else:
                st.write(iq)

def render_tailor_resume() -> None:
    """Render the tailor resume page."""
    st.header("✍️ Tailor Resume")
    st.write("Generate a tailored resume draft (uses stored resume + JD).")
    
    if not st.session_state.get(AppConfig.RESUME_TEXT) or not st.session_state.get(AppConfig.JOB_DESCRIPTION):
        st.info("Upload a resume and paste a job description on the Analyze Resume page first.")
    else:
        if st.button("Generate Tailored Resume"):
            try:
                with st.spinner("Generating tailored resume..."):
                    tailored = call_openai_tailor()
                st.session_state[AppConfig.TAILORED_RESUME] = tailored
            except Exception as e:
                st.error(f"Tailoring failed: {e}")
                logger.error(f"Tailoring error: {e}")

        if st.session_state.get(AppConfig.TAILORED_RESUME):
            st.subheader("Tailored Resume Draft")
            st.text_area("Tailored Resume", st.session_state[AppConfig.TAILORED_RESUME], height=420)

# ============================================================================
# MAIN APP LOGIC
# ============================================================================

def main() -> None:
    """Main application entry point."""
    page = render_sidebar()
    
    if page == AppConfig.PAGE_HOME:
        render_home()
    elif page == AppConfig.PAGE_ANALYZE:
        render_analyze_resume()
    elif page == AppConfig.PAGE_ATS:
        render_ats_check()
    elif page == AppConfig.PAGE_INTERVIEW:
        render_interview_prep()
    elif page == AppConfig.PAGE_TAILOR:
        render_tailor_resume()

if __name__ == "__main__":
    main()
