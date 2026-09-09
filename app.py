```python
import io
import json
import os
import re

import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openai/gpt-oss-20b"


# ============================================================
# ATS JSON SCHEMA
# ============================================================

ATS_SCHEMA = {
    "type": "object",
    "properties": {
        "ats_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "score_label": {
            "type": "string",
        },
        "summary": {
            "type": "string",
        },
        "category_scores": {
            "type": "object",
            "properties": {
                "structure_formatting": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 20,
                },
                "contact_summary": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                },
                "skills_keywords": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 20,
                },
                "experience": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 25,
                },
                "education_certifications": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                },
                "clarity_consistency": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                },
                "ats_risk_factors": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5,
                },
            },
            "required": [
                "structure_formatting",
                "contact_summary",
                "skills_keywords",
                "experience",
                "education_certifications",
                "clarity_consistency",
                "ats_risk_factors",
            ],
            "additionalProperties": False,
        },
        "strengths": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "improvements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "priority": {
                        "type": "string",
                    },
                    "issue": {
                        "type": "string",
                    },
                    "recommendation": {
                        "type": "string",
                    },
                },
                "required": [
                    "priority",
                    "issue",
                    "recommendation",
                ],
                "additionalProperties": False,
            },
        },
        "missing_keywords": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "formatting_risks": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "keyword_alignment": {
            "type": "string",
        },
        "section_feedback": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                },
                "skills": {
                    "type": "string",
                },
                "experience": {
                    "type": "string",
                },
                "education": {
                    "type": "string",
                },
            },
            "required": [
                "summary",
                "skills",
                "experience",
                "education",
            ],
            "additionalProperties": False,
        },
    },
    "required": [
        "ats_score",
        "score_label",
        "summary",
        "category_scores",
        "strengths",
        "improvements",
        "missing_keywords",
        "formatting_risks",
        "keyword_alignment",
        "section_feedback",
    ],
    "additionalProperties": False,
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert ATS resume analyzer and professional recruiter.

Your task is to analyze a resume conservatively and realistically.

IMPORTANT RULES:

1. Never invent information that is not present in the resume.
2. Never assume a skill, qualification, certification, job title,
   achievement, degree, or technology unless supported by the resume.
3. Give an ATS compatibility estimate, not a guarantee of getting hired.
4. Be objective and constructive.
5. Do not penalize a candidate simply because something is not present
   unless it is genuinely relevant to ATS performance.
6. Do not reward imaginary achievements.
7. Separate formatting/ATS issues from actual candidate quality.
8. If a job description is provided, compare the resume against it.
9. Identify important keywords from the job description that are missing
   from the resume.
10. Do not recommend adding a keyword unless the candidate actually has
    relevant experience with it.
11. Keep recommendations practical and specific.

ATS SCORE:

The total score must be 100 points.

1. Structure and formatting: 20 points
2. Contact information and summary: 10 points
3. Skills and keywords: 20 points
4. Professional experience: 25 points
5. Education and certifications: 10 points
6. Clarity and consistency: 10 points
7. ATS risk factors: 5 points

The final ats_score must equal the sum of the seven category scores.

SCORING GUIDANCE:

90-100 = Excellent ATS readiness
80-89  = Very good
70-79  = Good but improvements recommended
60-69  = Needs improvement
Below 60 = Significant ATS improvements needed

Consider common ATS risks such as:

- tables
- columns
- text boxes
- headers/footers containing important information
- graphics
- icons replacing text
- unusual symbols
- excessive formatting
- inconsistent dates
- inconsistent job titles
- missing contact information
- poor section headings
- keyword stuffing
- overly long paragraphs
- unclear work history
- missing measurable achievements

For job matching:

- Identify important keywords present in the job description.
- Identify relevant keywords already present in the resume.
- Identify potentially missing keywords.
- Do not tell the candidate to add a keyword if the resume provides
  no evidence that they have that skill.

Return ONLY valid JSON matching the provided schema.
"""


# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_pdf_text(file_bytes):
    """Extract text from a PDF."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))

        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        return "\n".join(pages).strip()

    except Exception as exc:
        raise RuntimeError(
            f"Could not read PDF: {exc}"
        ) from exc


def extract_docx_text(file_bytes):
    """Extract paragraphs and tables from DOCX."""
    try:
        document = Document(io.BytesIO(file_bytes))

        parts = []

        # Paragraphs
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()

            if text:
                parts.append(text)

        # Tables
        for table in document.tables:
            for row in table.rows:
                cells = []

                for cell in row.cells:
                    cell_text = cell.text.strip()

                    if cell_text:
                        cells.append(cell_text)

                if cells:
                    parts.append(" | ".join(cells))

        return "\n".join(parts).strip()

    except Exception as exc:
        raise RuntimeError(
            f"Could not read DOCX: {exc}"
        ) from exc


def extract_txt_text(file_bytes):
    """Extract text from TXT."""
    try:
        return file_bytes.decode(
            "utf-8",
            errors="ignore"
        ).strip()

    except Exception as exc:
        raise RuntimeError(
            f"Could not read TXT file: {exc}"
        ) from exc


def extract_resume_text(uploaded_file):
    """Detect file type and extract resume text."""

    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_pdf_text(file_bytes)

    if filename.endswith(".docx"):
        return extract_docx_text(file_bytes)

    if filename.endswith(".txt"):
        return extract_txt_text(file_bytes)

    raise RuntimeError(
        "Unsupported file type. Please upload PDF, DOCX, or TXT."
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """Clean excessive whitespace while preserving useful structure."""

    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# GROQ API
# ============================================================

def get_groq_api_key():
    """Get Groq API key from Streamlit secrets or environment."""

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    return api_key


def analyze_resume(
    resume_text,
    job_description="",
):
    """Send resume to Groq and return structured ATS analysis."""

    api_key = get_groq_api_key()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add GROQ_API_KEY to Streamlit Secrets."
        )

    client = Groq(api_key=api_key)

    job_text = job_description.strip()

    if not job_text:
        job_text = (
            "No job description was provided. "
            "Analyze the resume for general ATS readiness."
        )

    user_prompt = f"""
RESUME:

{resume_text}


JOB DESCRIPTION:

{job_text}


Analyze the resume according to your instructions.

Return only the JSON object matching the ATS schema.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ats_analysis",
                    "strict": True,
                    "schema": ATS_SCHEMA,
                },
            },
        )

    except Exception as exc:

        error_text = str(exc)

        if "429" in error_text or "rate_limit" in error_text.lower():
            raise RuntimeError(
                "Groq rate limit reached. "
                "Please wait and try again later."
            ) from exc

        if "401" in error_text or "authentication" in error_text.lower():
            raise RuntimeError(
                "Groq API key is invalid or not configured correctly."
            ) from exc

        if "model" in error_text.lower() and (
            "not found" in error_text.lower()
            or "does not exist" in error_text.lower()
        ):
            raise RuntimeError(
                f"The Groq model '{MODEL}' is unavailable. "
                "Check the model name in the application."
            ) from exc

        raise RuntimeError(
            f"Groq API error: {type(exc).__name__}: {exc}"
        ) from exc

    if not response.choices:
        raise RuntimeError(
            "Groq returned no response choices."
        )

    output_text = response.choices[0].message.content

    if not output_text:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    try:
        result = json.loads(output_text)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Groq returned invalid JSON."
        ) from exc

    return result


# ============================================================
# SCORE DISPLAY
# ============================================================

def score_label(score):
    if score >= 90:
        return "Excellent"

    if score >= 80:
        return "Very Good"

    if score >= 70:
        return "Good"

    if score >= 60:
        return "Needs Improvement"

    return "Needs Significant Improvement"


def display_category_scores(category_scores):

    st.subheader("📊 Category Scores")

    categories = [
        (
            "Structure & Formatting",
            "structure_formatting",
            20,
        ),
        (
            "Contact & Summary",
            "contact_summary",
            10,
        ),
        (
            "Skills & Keywords",
            "skills_keywords",
            20,
        ),
        (
            "Experience",
            "experience",
            25,
        ),
        (
            "Education & Certifications",
            "education_certifications",
            10,
        ),
        (
            "Clarity & Consistency",
            "clarity_consistency",
            10,
        ),
        (
            "ATS Risk Factors",
            "ats_risk_factors",
            5,
        ),
    ]

    cols = st.columns(4)

    for index, (label, key, maximum) in enumerate(categories):

        value = category_scores.get(key, 0)

        col = cols[index % 4]

        with col:
            st.metric(
                label,
                f"{value}/{maximum}",
            )


# ============================================================
# RESULTS DISPLAY
# ============================================================

def display_results(result):

    # --------------------------------------------------------
    # Overall score
    # --------------------------------------------------------

    score = int(result.get("ats_score", 0))

    label = result.get(
        "score_label",
        score_label(score),
    )

    st.divider()

    st.subheader("🎯 ATS Score")

    score_col, summary_col = st.columns([1, 2])

    with score_col:
        st.metric(
            "Overall Score",
            f"{score}/100",
        )

        st.progress(
            max(0, min(score, 100)) / 100
        )

        st.caption(label)

    with summary_col:
        st.markdown("### Summary")
        st.write(
            result.get(
                "summary",
                "No summary available.",
            )
        )

    # --------------------------------------------------------
    # Category scores
    # --------------------------------------------------------

    category_scores = result.get(
        "category_scores",
        {},
    )

    display_category_scores(category_scores)

    # --------------------------------------------------------
    # Strengths
    # --------------------------------------------------------

    st.divider()

    st.subheader("✅ Strengths")

    strengths = result.get(
        "strengths",
        [],
    )

    if strengths:

        for strength in strengths:
            st.success(strength)

    else:
        st.info("No specific strengths were identified.")

    # --------------------------------------------------------
    # Missing keywords
    # --------------------------------------------------------

    st.subheader("🔑 Missing Keywords")

    missing_keywords = result.get(
        "missing_keywords",
        [],
    )

    if missing_keywords:

        keyword_text = ", ".join(
            missing_keywords
        )

        st.warning(keyword_text)

    else:
        st.success(
            "No major missing keywords were identified."
        )

    # --------------------------------------------------------
    # Keyword alignment
    # --------------------------------------------------------

    st.subheader("🔎 Keyword Alignment")

    st.write(
        result.get(
            "keyword_alignment",
            "No keyword alignment analysis available.",
        )
    )

    # --------------------------------------------------------
    # Formatting risks
    # --------------------------------------------------------

    st.subheader("⚠️ Formatting & ATS Risks")

    formatting_risks = result.get(
        "formatting_risks",
        [],
    )

    if formatting_risks:

        for risk in formatting_risks:
            st.warning(risk)

    else:
        st.success(
            "No significant ATS formatting risks identified."
        )

    # --------------------------------------------------------
    # Improvements
    # --------------------------------------------------------

    st.divider()

    st.subheader("🛠️ Recommended Improvements")

    improvements = result.get(
        "improvements",
        [],
    )

    if improvements:

        for improvement in improvements:

            priority = improvement.get(
                "priority",
                "Medium",
            )

            issue = improvement.get(
                "issue",
                "",
            )

            recommendation = improvement.get(
                "recommendation",
                "",
            )

            with st.expander(
                f"{priority}: {issue}"
            ):

                st.write(
                    recommendation
                )

    else:
        st.info(
            "No specific improvements were identified."
        )

    # --------------------------------------------------------
    # Section feedback
    # --------------------------------------------------------

    st.divider()

    st.subheader("📝 Section-by-Section Feedback")

    feedback = result.get(
        "section_feedback",
        {},
    )

    tabs = st.tabs(
        [
            "Summary",
            "Skills",
            "Experience",
            "Education",
        ]
    )

    with tabs[0]:
        st.write(
            feedback.get(
                "summary",
                "No feedback available.",
            )
        )

    with tabs[1]:
        st.write(
            feedback.get(
                "skills",
                "No feedback available.",
            )
        )

    with tabs[2]:
        st.write(
            feedback.get(
                "experience",
                "No feedback available.",
            )
        )

    with tabs[3]:
        st.write(
            feedback.get(
                "education",
                "No feedback available.",
            )
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📄 Resume ATS Analyzer")

    st.markdown(
        """
### About

Analyze your resume for:

- ATS compatibility
- Keywords
- Formatting risks
- Experience
- Education
- Skills
- Job-description alignment
"""
    )

    st.divider()

    st.markdown("### 🤖 AI Model")

    st.code(
        MODEL,
        language="text",
    )

    st.caption(
        "Powered by Groq"
    )

    st.divider()

    st.markdown("### 🔒 Privacy")

    st.caption(
        "Your uploaded resume is processed for "
        "the current analysis and is not stored "
        "by this application."
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

st.title("📄 Resume ATS Analyzer")

st.markdown(
    """
Upload your resume and get an ATS-focused analysis.
You can optionally provide a job description to check
keyword alignment and job matching.
"""
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=[
        "pdf",
        "docx",
        "txt",
    ],
    help="Supported formats: PDF, DOCX, TXT",
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

job_description = st.text_area(
    "Job Description (Optional)",
    height=220,
    placeholder=(
        "Paste the job description here for "
        "keyword and job-match analysis..."
    ),
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if not uploaded_file:

        st.error(
            "Please upload a resume first."
        )

        st.stop()

    try:

        with st.spinner(
            "Extracting resume text..."
        ):

            resume_text = extract_resume_text(
                uploaded_file
            )

            resume_text = clean_text(
                resume_text
            )

        if not resume_text:

            st.error(
                "No readable text was found in the uploaded file."
            )

            st.info(
                "If this is a scanned/image-only PDF, "
                "OCR may be required."
            )

            st.stop()

        # Safety limit for extremely large resumes
        MAX_CHARS = 60000

        if len(resume_text) > MAX_CHARS:

            resume_text = resume_text[:MAX_CHARS]

            st.warning(
                "The resume was very large, so the text "
                "was limited to the first 60,000 characters."
            )

        with st.spinner(
            "Analyzing resume with Groq AI..."
        ):

            result = analyze_resume(
                resume_text=resume_text,
                job_description=job_description,
            )

        st.session_state["ats_result"] = result

        st.success(
            "Analysis completed successfully!"
        )

    except Exception as exc:

        st.error(
            f"Analysis failed: {exc}"
        )

        error_text = str(exc).lower()

        if "rate limit" in error_text:

            st.warning(
                "Groq's free-tier rate limit has been reached. "
                "Please wait before trying again."
            )

        elif "api key" in error_text:

            st.info(
                "Check your GROQ_API_KEY in "
                "Streamlit Cloud → Settings → Secrets."
            )


# ============================================================
# SHOW STORED RESULTS
# ============================================================

if "ats_result" in st.session_state:

    display_results(
        st.session_state["ats_result"]
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Resume ATS Analyzer • AI-assisted resume evaluation • "
    "Always verify recommendations against the actual job requirements."
)
```
