def build_analyze_prompt(resume_text: str, jd_text: str) -> str:
    return f"""
You are an ATS (Applicant Tracking System). Compare the following resume against the job description.

Resume:
{resume_text}

Job Description:
{jd_text}

Provide:
1. Overall % match score.
2. Section-wise scores (Skills, Experience, Education).
3. Key strengths.
4. Missing keywords.
    """

def build_tailor_prompt(resume_text: str, jd_text: str) -> str:
    return f"""
You are a professional resume writer. Rewrite and tailor the resume below to fit the job description.

Resume:
{resume_text}

Job Description:
{jd_text}

Make the resume ATS-friendly, concise, and aligned to the role.
    """

def build_interview_prep_prompt(resume_text: str, jd_text: str, num_questions: int = 5) -> str:
    return f"""
Based on the resume and job description, generate {num_questions} possible interview questions 
that the candidate should prepare for. Include both technical and behavioral questions.

Resume:
{resume_text}

Job Description:
{jd_text}
    """
