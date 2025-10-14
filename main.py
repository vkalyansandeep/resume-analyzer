import streamlit as st
import fitz  # PyMuPDF
import docx
import re
import os
import json
import hashlib
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from io import BytesIO

st.set_page_config(page_title="Resume Analyzer & Tailor", layout="wide")
st.title("🧭 LLM-Powered Resume Analyzer & Tailor")

def extract_text(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        return "".join(page.get_text() for page in pdf)
    elif name.endswith(".docx"):
        d = docx.Document(file)
        return "\n".join(p.text for p in d.paragraphs)
    else:
        return ""

def clamp_int(n, lo=0, hi=100):
    try:
        return max(lo, min(hi, int(n)))
    except Exception:
        return None

def score_color(score:int):
    if score is None:
        return "gray"
    if score >= 70:
        return "green"
    if score >= 50:
        return "orange"
    return "red"

def render_score_big_half(col, label:str, score:int):
    color = score_color(score)
    col.markdown(
        f"<div style='font-size:40px; font-weight:800; color:{color};'>{label}: {score}%</div>",
        unsafe_allow_html=True,
    )
    col.markdown(
        f"""
        <div style="background-color:#e6e6e6; border-radius:10px; height:20px; width:100%; margin:6px 0 12px 0;">
            <div style="background-color:{color}; width:{score}%; height:100%; border-radius:10px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_ats_big(col, label:str, score:int):
    if score is None:
        color = "gray"
    elif score >= 80:
        color = "green"
    elif score >= 60:
        color = "orange"
    else:
        color = "red"
    col.markdown(
        f"<div style='font-size:40px; font-weight:800; color:{color};'>{label}: {score}%</div>",
        unsafe_allow_html=True,
    )

def try_parse_json_from_text(text:str):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return None

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
ATS_CACHE_FILE = CACHE_DIR / "ats_cache.json"

def _load_ats_cache():
    if ATS_CACHE_FILE.exists():
        try:
            return json.loads(ATS_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_ats_cache(obj: dict):
    ATS_CACHE_FILE.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def resume_hash(text:str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def ats_get_cached(h:str):
    obj = _load_ats_cache()
    return obj.get(h)

def ats_save_cached(h:str, data:dict):
    obj = _load_ats_cache()
    obj[h] = data
    obj[h]["cached_at"] = datetime.utcnow().isoformat()
    _save_ats_cache(obj)

def run_structured_analysis():
    client = OpenAI()
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
{st.session_state.get("resume_text","")}

JOB_DESCRIPTION:
{st.session_state.get("job_description","")}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.12,
        max_tokens=1200,
    )
    return resp.choices[0].message.content

def call_openai_tailor():
    client = OpenAI()
    prompt = f"""
You are an expert resume writer. Using the RESUME and JOB_DESCRIPTION below, produce a tailored resume draft that:
- keeps factual content from RESUME (do not invent new jobs or dates),
- emphasizes experiences and keywords relevant to JOB_DESCRIPTION,
- outputs ONLY the revised resume text (no extra commentary).

RESUME:
{st.session_state.get("resume_text","")}

JOB_DESCRIPTION:
{st.session_state.get("job_description","")}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.28,
        max_tokens=1200,
    )
    return resp.choices[0].message.content

def call_openai_ats(resume_text:str):
    client = OpenAI()
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
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.12,
        max_tokens=700,
    )
    parsed = try_parse_json_from_text(resp.choices[0].message.content)
    return parsed if parsed else {"overall_score": None, "issues": [], "summary": resp.choices[0].message.content}

def call_openai_interview(num_questions:int):
    client = OpenAI()
    prompt = f"""
You are an interviewer and career coach. Using the JOB_DESCRIPTION and the RESUME below:
- Generate {num_questions} role-specific interview questions.
- For each question include a short suggested bullet-answer derived from the RESUME (do NOT invent facts).
- Output JSON:
{{ "questions": [ {{"q":"<question>", "suggested_answer":"<short answer>"}} ] }}

JOB_DESCRIPTION:
{st.session_state.get("job_description","")}

RESUME:
{st.session_state.get("resume_text","")}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.25,
        max_tokens=1000,
    )
    parsed = try_parse_json_from_text(resp.choices[0].message.content)
    return parsed if parsed else {"questions": []}

def generate_report_bytes():
    title = "Resume Analysis Report"
    sections = {}
    aj = st.session_state.get("analysis_json")
    ms = st.session_state.get("match_score")
    sections["Overall Match Score"] = f"{ms}%" if ms is not None else "N/A"
    if aj and isinstance(aj, dict):
        sec_lines = []
        for sname, sdata in aj.get("sections", {}).items():
            score = sdata.get("score", "N/A")
            comment = sdata.get("comment", "")
            sec_lines.append(f"{sname.title()}: {score}%\n{comment}")
        sections["Section-wise Breakdown"] = "\n\n".join(sec_lines)
        sections["Matches"] = "\n".join(aj.get("matches", [])) if aj.get("matches") else "N/A"
        missing = aj.get("missing_keywords", [])
        if isinstance(missing, list) and missing and isinstance(missing[0], dict):
            sections["Missing Keywords"] = "\n".join([f"{m.get('keyword')}: {m.get('reason')}" for m in missing])
        else:
            sections["Missing Keywords"] = ", ".join(missing) if missing else "N/A"
        tools = aj.get("recommended_tools", [])
        if isinstance(tools, list) and tools and isinstance(tools[0], dict):
            sections["Recommended Tools"] = "\n".join([f"{t.get('tool')}: {t.get('reason')}" for t in tools])
        else:
            sections["Recommended Tools"] = ", ".join(tools) if tools else "N/A"
        recs = aj.get("recommendations", [])
        sections["Recommendations"] = "\n".join(recs) if recs else "N/A"
        sections["Full Assessment"] = aj.get("full_assessment", "") or st.session_state.get("analysis_raw","")
    else:
        sections["Analysis Raw"] = st.session_state.get("analysis_raw","")

    if st.session_state.get("tailored_resume"):
        sections["Tailored Resume"] = st.session_state.get("tailored_resume")

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
    except Exception:
        text = [title, "\n"]
        for heading, content in sections.items():
            text.append(heading)
            text.append(str(content))
            text.append("\n\n")
        return "\n".join(text).encode("utf-8")

for k,v in {
    "resume_text": None,
    "job_description": None,
    "analysis_raw": None,
    "analysis_json": None,
    "match_score": None,
    "tailored_resume": None,
    "ats_report": None,
    "interview_questions": None,
    "last_filename": None,
    "page": "Home",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


st.sidebar.title("Navigation")
if st.sidebar.button("🏠 Home"):
    st.session_state["page"] = "Home"
if st.sidebar.button("📊 Analyze Resume"):
    st.session_state["page"] = "Analyze Resume"
if st.sidebar.button("✍️ Tailor Resume"):
    st.session_state["page"] = "Tailor Resume"
if st.sidebar.button("✅ ATS Check"):
    st.session_state["page"] = "ATS Check"
if st.sidebar.button("🎤 Interview Prep"):
    st.session_state["page"] = "Interview Prep"

st.sidebar.markdown("<div style='height:220px'></div>", unsafe_allow_html=True)

if st.sidebar.button("🔄 Restart Session"):
    keys = [
        "resume_text",
        "job_description",
        "analysis_raw",
        "analysis_json",
        "match_score",
        "tailored_resume",
        "ats_report",
        "interview_questions",
        "last_filename",
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["page"] = "Home"
    try:
        st.experimental_rerun()
    except Exception:
        st.markdown("<script>window.location.reload()</script>", unsafe_allow_html=True)

page = st.session_state.get("page", "Home")

if page == "Home":
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

elif page == "Analyze Resume":
    st.header("📊 Analyze Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    job_description = st.text_area("Paste the Job Description here", height=220, placeholder="Paste full JD text here...")

    if uploaded_file is not None:
        if st.session_state.get("last_filename") != uploaded_file.name:
            st.session_state["resume_text"] = extract_text(uploaded_file)
            st.session_state["analysis_raw"] = None
            st.session_state["analysis_json"] = None
            st.session_state["match_score"] = None
            st.session_state["tailored_resume"] = None
            st.session_state["ats_report"] = None
            st.session_state["interview_questions"] = None
            st.session_state["last_filename"] = uploaded_file.name

    if job_description and job_description.strip():
        st.session_state["job_description"] = job_description

    if st.button("Analyze Resume"):
        if not st.session_state.get("resume_text") or not st.session_state.get("job_description"):
            st.error("Please upload a resume and paste the job description first.")
        else:
            with st.spinner("Analyzing (structured)..."):
                raw = run_structured_analysis()
            st.session_state["analysis_raw"] = raw
            parsed = try_parse_json_from_text(raw)
            st.session_state["analysis_json"] = parsed
            if parsed and isinstance(parsed, dict):
                st.session_state["match_score"] = clamp_int(parsed.get("match_score"))
            else:
                m = re.search(r"(\d{1,3})\s*%", raw)
                st.session_state["match_score"] = clamp_int(m.group(1)) if m else None

    if st.session_state.get("analysis_json"):
        aj = st.session_state["analysis_json"]
        score = st.session_state.get("match_score")

        col_left, col_right = st.columns([1,1])
        if score is not None:
            render_score_big_half(col_left, "Match Score", score)
        if aj.get("quick_summary"):
            col_right.markdown("**Quick summary**")
            col_right.write(aj.get("quick_summary"))

        st.subheader("Section-wise breakdown")
        sections = aj.get("sections", {})
        ordered = [("summary","Summary"), ("skills","Skills"), ("experience","Experience"), ("education","Education")]
        for i in range(0, len(ordered), 2):
            a_key, a_label = ordered[i]
            b_key, b_label = ordered[i+1]
            col_a, col_b = st.columns(2)
            a_data = sections.get(a_key, {}) if sections else {}
            b_data = sections.get(b_key, {}) if sections else {}
            a_score = clamp_int(a_data.get("score")) if a_data else None
            b_score = clamp_int(b_data.get("score")) if b_data else None
            if a_score is not None:
                col_a.write(f"**{a_label}** — {a_score}%")
                col_a.progress(a_score/100.0)
                if a_data.get("comment"):
                    col_a.caption(a_data.get("comment"))
            else:
                col_a.write(f"**{a_label}** — N/A")
            if b_score is not None:
                col_b.write(f"**{b_label}** — {b_score}%")
                col_b.progress(b_score/100.0)
                if b_data.get("comment"):
                    col_b.caption(b_data.get("comment"))
            else:
                col_b.write(f"**{b_label}** — N/A")

        st.subheader("Detailed analysis & suggestions")
        if aj.get("matches"):
            st.markdown("**What in your resume already matches the JD:**")
            st.write(", ".join(aj.get("matches")))
        if aj.get("missing_keywords"):
            st.markdown("**Missing keywords / skills (consider adding these):**")
            missing = aj.get("missing_keywords")
            if isinstance(missing, list) and missing and isinstance(missing[0], dict):
                for m in missing:
                    st.write(f"- **{m.get('keyword')}** — {m.get('reason','')}")
            else:
                st.write(", ".join(missing))
        if aj.get("recommended_tools"):
            st.markdown("**Recommended tools / technologies to highlight:**")
            tools = aj.get("recommended_tools")
            if isinstance(tools, list) and tools and isinstance(tools[0], dict):
                for t in tools:
                    st.write(f"- **{t.get('tool')}** — {t.get('reason','')}")
            else:
                st.write(", ".join(tools))
        if aj.get("recommendations"):
            st.markdown("**Actionable recommendations (prioritized):**")
            for r in aj.get("recommendations"):
                st.write(f"- {r}")

        st.markdown("---")
        st.markdown("**Full assessment**")
        if aj.get("full_assessment"):
            st.write(aj.get("full_assessment"))
        else:
            st.write(st.session_state.get("analysis_raw",""))

    elif st.session_state.get("analysis_raw"):
        st.info("Analysis in progress or raw output available:")
        st.write(st.session_state.get("analysis_raw"))

elif page == "ATS Check":
    st.header("✅ ATS Friendliness Check")
    st.write("Evaluates ATS compatibility for the uploaded resume (uses stored resume).")
    if not st.session_state.get("resume_text"):
        st.info("Upload a resume first on the Analyze Resume page to run ATS checks.")
    else:
        if st.button("Run ATS Check"):
            text = st.session_state.get("resume_text")
            h = resume_hash(text)
            cached = ats_get_cached(h)
            if cached:
                st.success("Using cached ATS result (resume unchanged).")
                result = cached
            else:
                with st.spinner("Running ATS check..."):
                    result = call_openai_ats(text)
                ats_save_cached(h, result)
            st.session_state["ats_report"] = result

        if st.session_state.get("ats_report"):
            ar = st.session_state["ats_report"]
            overall = clamp_int(ar.get("overall_score")) if isinstance(ar, dict) else None
            if overall is not None:
                # large colored ATS score (no progress bar)
                render_ats_big(st, "ATS Score", overall)
            st.subheader("Issues & Recommendations")
            if isinstance(ar, dict):
                for it in ar.get("issues", []):
                    st.write(f"- **{it.get('problem','Issue')}** — {it.get('recommendation','')}")
                st.markdown("---")
                st.write(ar.get("summary",""))
            else:
                st.write(ar)

elif page == "Interview Prep":
    st.header("🎤 Interview Prep")
    st.write("Generate tailored interview questions & suggested answers (uses stored resume + JD).")
    if not st.session_state.get("resume_text") or not st.session_state.get("job_description"):
        st.info("Upload a resume and paste a job description on the Analyze Resume page first.")
    else:
        # determine number of questions from JD length and match score
        jd = st.session_state.get("job_description","")
        jd_len = len(jd.split())
        match = st.session_state.get("match_score") or 0
        # heuristic: base 6, add up to 4 more depending on JD length and match score
        extra = 0
        if jd_len > 600:
            extra += 2
        elif jd_len > 300:
            extra += 1
        if match >= 80:
            extra += 2
        elif match >= 60:
            extra += 1
        num_questions = min(10, 6 + extra)

        if st.button("Generate Interview Questions"):
            with st.spinner("Generating interview questions..."):
                q_json = call_openai_interview(num_questions)
            st.session_state["interview_questions"] = q_json

        if st.session_state.get("interview_questions"):
            iq = st.session_state["interview_questions"]
            if isinstance(iq, dict) and iq.get("questions"):
                for idx, q in enumerate(iq.get("questions", []), start=1):
                    st.markdown(f"**Q{idx}: {q.get('q')}**")
                    st.markdown(f"- Suggested answer: {q.get('suggested_answer')}")
                    st.markdown("---")
            else:
                st.write(iq)

elif page == "Tailor Resume":
    st.header("✍️ Tailor Resume")
    st.write("Generate a tailored resume draft (uses stored resume + JD).")
    if not st.session_state.get("resume_text") or not st.session_state.get("job_description"):
        st.info("Upload a resume and paste a job description on the Analyze Resume page first.")
    else:
        if st.button("Generate Tailored Resume"):
            with st.spinner("Generating tailored resume..."):
                tailored = call_openai_tailor()
            st.session_state["tailored_resume"] = tailored

        if st.session_state.get("tailored_resume"):
            st.subheader("Tailored Resume Draft")
            st.text_area("Tailored Resume", st.session_state["tailored_resume"], height=420)
