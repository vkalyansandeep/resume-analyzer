"""
Centralized prompt templates for the Resume Analyzer project.
"""

def build_analyze_prompt(resume_text, job_description):
    return f"""
You are an expert resume analyst. Compare the RESUME and the JOB DESCRIPTION
and respond in exactly this format:

Match Score: <number>%    <-- number only, 0-100
Quick Summary:
- Very short overview (2–3 bullet points)

Full Assessment:
- Detailed analysis of alignment between resume and job description
- Highlight strengths
- Highlight weaknesses
- Provide thoughtful insights (not repetitive)

Missing Keywords / Skills:
- List keywords or skills from the JD missing in the resume

Recommended Tools / Technologies:
- Suggest relevant tools/technologies to emphasize

Actionable Recommendations:
- Clear, practical steps for improving the resume

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}
    """.strip()


def build_tailor_prompt(resume_text, job_description):
    return f"""
You are an expert career coach and resume writer. Rewrite the RESUME to align with the JOB DESCRIPTION.
Rules:
- Keep it truthful: do NOT invent roles, companies, or achievements.
- Emphasize relevant experience and skills.
- Naturally incorporate missing keywords where appropriate.
- Keep a professional tone and clean, ATS-friendly formatting.
- Output ONLY the revised resume content.

RESUME (original):
{resume_text}

JOB DESCRIPTION:
{job_description}
    """.strip()


def build_ats_prompt(resume_text):
    return f"""
You are an ATS (Applicant Tracking System) evaluator.
Analyze the following RESUME for formatting, structure, keyword optimization, and readability.

Provide the output in this structure:

ATS Score: <number>%    <-- number only, 0-100
Strengths:
- ...
- ...
Weaknesses:
- ...
- ...
Recommendations:
- ...

RESUME:
{resume_text}
    """.strip()


def build_interview_prompt(resume_text, job_description, num_questions=6):
    return f"""
You are an expert interviewer. Generate {num_questions} tailored interview questions 
based on BOTH the RESUME and the JOB DESCRIPTION. 
Focus on role-specific, technical, and behavioral questions that help assess readiness.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}
    """.strip()
