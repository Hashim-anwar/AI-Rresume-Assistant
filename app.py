import io
import json
import os
import re

import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openai/gpt-oss-20b"


# ============================================================
# ATS SCHEMA
# ============================================================

ATS_SCHEMA = {
    "ats_score": 0,
    "score_label": "",
    "summary": "",
    "category_scores": {
        "structure_formatting": 0,
        "contact_summary": 0,
        "skills_keywords": 0,
        "experience": 0,
        "education_certifications": 0,
        "clarity_consistency": 0,
        "ats_risk_factors": 0
    },
    "strengths": [],
    "improvements": [],
    "missing_keywords": [],
    "formatting_risks": [],
    "keyword_alignment": "",
    "section_feedback": {
        "summary": "",
        "skills": "",
        "experience": "",
        "education": ""
    }
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert ATS resume analyzer, professional recruiter,
career consultant, and resume optimization specialist.

Your job is to analyze the candidate's resume realistically.

IMPORTANT RULES:

1. Never invent information.
2. Never assume the candidate has a skill that is not supported
   by the resume.
3. Never invent qualifications, certifications, degrees,
   technologies, achievements, responsibilities, job titles,
   employers, dates, or experience.
4. Do not create fake achievements.
5. Give an ATS compatibility estimate, not a guarantee of getting
   an interview or job.
6. Be objective, practical, and constructive.
7. Separate ATS formatting problems from candidate quality.
8. If a job description is provided, compare the resume against it.
9. Identify important keywords in the job description.
10. Identify keywords that appear to be missing from the resume.
11. Do not recommend adding a keyword unless the candidate has
    genuine relevant experience with that skill.
12. Never encourage keyword stuffing.
13. Give practical recommendations that the candidate can actually use.

ATS SCORE:

The final ATS score is out of 100.

Structure and Formatting = 20 points
Contact Information and Summary = 10 points
Skills and Keywords = 20 points
Professional Experience = 25 points
Education and Certifications = 10 points
Clarity and Consistency = 10 points
ATS Risk Factors = 5 points

The seven category scores MUST add up exactly to the final ATS score.

CATEGORY MAXIMUMS:

structure_formatting: maximum 20
contact_summary: maximum 10
skills_keywords: maximum 20
experience: maximum 25
education_certifications: maximum 10
clarity_consistency: maximum 10
ats_risk_factors: maximum 5

SCORE LABELS:

90-100 = Excellent ATS Readiness
80-89 = Very Good ATS Readiness
70-79 = Good ATS Readiness
60-69 = Needs Improvement
0-59 = Significant Improvement Needed

CHECK FOR ATS RISKS INCLUDING:

- Tables
- Multiple columns
- Text boxes
- Images
- Graphics
- Icons replacing text
- Important information inside headers
- Important information inside footers
- Unusual symbols
- Excessive formatting
- Inconsistent dates
- Inconsistent job titles
- Missing contact information
- Poor section headings
- Keyword stuffing
- Long paragraphs
- Unclear employment history
- Missing measurable achievements
- Poor readability
- Unnecessary personal information

JOB DESCRIPTION ANALYSIS:

If a job description is provided:

1. Identify important technical keywords.
2. Identify important soft-skill keywords.
3. Identify relevant industry terms.
4. Identify keywords already present in the resume.
5. Identify important keywords missing from the resume.
6. Explain the overall alignment.
7. Do not claim the candidate has experience they do not have.

OUTPUT:

Return ONLY ONE valid JSON object.

Do not use Markdown.
Do not use ```json.
Do not write any explanation before or after the JSON.

The JSON must contain these fields:

ats_score
score_label
summary
category_scores
strengths
improvements
missing_keywords
formatting_risks
keyword_alignment
section_feedback

category_scores must contain:

structure_formatting
contact_summary
skills_keywords
experience
education_certifications
clarity_consistency
ats_risk_factors

improvements must be an array of objects.

Every improvement object must contain:

priority
issue
recommendation

section_feedback must contain:

summary
skills
experience
education

Make the response concise enough to fit within the model's output limits.
"""


# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_pdf_text(file_bytes):

    try:

        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages).strip()

    except Exception as exc:

        raise RuntimeError(
            f"Could not read PDF: {exc}"
        ) from exc


def extract_docx_text(file_bytes):

    try:

        document = Document(
            io.BytesIO(file_bytes)
        )

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

                    text = cell.text.strip()

                    if text:
                        cells.append(text)

                if cells:

                    parts.append(
                        " | ".join(cells)
                    )

        return "\n".join(parts).strip()

    except Exception as exc:

        raise RuntimeError(
            f"Could not read DOCX: {exc}"
        ) from exc


def extract_txt_text(file_bytes):

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

    file_bytes = uploaded_file.getvalue()

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):

        return extract_pdf_text(
            file_bytes
        )

    if filename.endswith(".docx"):

        return extract_docx_text(
            file_bytes
        )

    if filename.endswith(".txt"):

        return extract_txt_text(
            file_bytes
        )

    raise RuntimeError(
        "Unsupported file type. "
        "Please upload PDF, DOCX, or TXT."
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " "
    )

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# GET GROQ API KEY
# ============================================================

def get_groq_api_key():

    api_key = None

    try:

        api_key = st.secrets.get(
            "GROQ_API_KEY"
        )

    except Exception:

        pass

    if not api_key:

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

    return api_key


# ============================================================
# SCORE LABEL
# ============================================================

def get_score_label(score):

    if score >= 90:
        return "Excellent ATS Readiness"

    if score >= 80:
        return "Very Good ATS Readiness"

    if score >= 70:
        return "Good ATS Readiness"

    if score >= 60:
        return "Needs Improvement"

    return "Significant Improvement Needed"


# ============================================================
# VALIDATE AND NORMALIZE AI RESPONSE
# ============================================================

def normalize_result(result):

    if not isinstance(result, dict):

        raise RuntimeError(
            "AI returned an invalid response format."
        )

    # --------------------------------------------------------
    # Category scores
    # --------------------------------------------------------

    category_defaults = {
        "structure_formatting": 0,
        "contact_summary": 0,
        "skills_keywords": 0,
        "experience": 0,
        "education_certifications": 0,
        "clarity_consistency": 0,
        "ats_risk_factors": 0
    }

    categories = result.get(
        "category_scores",
        {}
    )

    if not isinstance(categories, dict):

        categories = {}

    maximums = {
        "structure_formatting": 20,
        "contact_summary": 10,
        "skills_keywords": 20,
        "experience": 25,
        "education_certifications": 10,
        "clarity_consistency": 10,
        "ats_risk_factors": 5
    }

    cleaned_categories = {}

    for key, default in category_defaults.items():

        value = categories.get(
            key,
            default
        )

        try:

            value = int(value)

        except Exception:

            value = default

        value = max(
            0,
            min(
                value,
                maximums[key]
            )
        )

        cleaned_categories[key] = value

    # --------------------------------------------------------
    # Calculate score ourselves
    # --------------------------------------------------------

    calculated_score = sum(
        cleaned_categories.values()
    )

    result["ats_score"] = calculated_score

    result["score_label"] = get_score_label(
        calculated_score
    )

    result["category_scores"] = (
        cleaned_categories
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    result["summary"] = str(
        result.get(
            "summary",
            "No summary available."
        )
    )

    # --------------------------------------------------------
    # Strengths
    # --------------------------------------------------------

    strengths = result.get(
        "strengths",
        []
    )

    if not isinstance(
        strengths,
        list
    ):

        strengths = []

    result["strengths"] = [
        str(item)
        for item in strengths
    ]

    # --------------------------------------------------------
    # Missing keywords
    # --------------------------------------------------------

    missing_keywords = result.get(
        "missing_keywords",
        []
    )

    if not isinstance(
        missing_keywords,
        list
    ):

        missing_keywords = []

    result["missing_keywords"] = [
        str(item)
        for item in missing_keywords
    ]

    # --------------------------------------------------------
    # Formatting risks
    # --------------------------------------------------------

    formatting_risks = result.get(
        "formatting_risks",
        []
    )

    if not isinstance(
        formatting_risks,
        list
    ):

        formatting_risks = []

    result["formatting_risks"] = [
        str(item)
        for item in formatting_risks
    ]

    # --------------------------------------------------------
    # Improvements
    # --------------------------------------------------------

    improvements = result.get(
        "improvements",
        []
    )

    if not isinstance(
        improvements,
        list
    ):

        improvements = []

    cleaned_improvements = []

    for item in improvements:

        if not isinstance(
            item,
            dict
        ):

            continue

        cleaned_improvements.append(
            {
                "priority": str(
                    item.get(
                        "priority",
                        "Medium"
                    )
                ),
                "issue": str(
                    item.get(
                        "issue",
                        ""
                    )
                ),
                "recommendation": str(
                    item.get(
                        "recommendation",
                        ""
                    )
                )
            }
        )

    result["improvements"] = (
        cleaned_improvements
    )

    # --------------------------------------------------------
    # Keyword alignment
    # --------------------------------------------------------

    result["keyword_alignment"] = str(
        result.get(
            "keyword_alignment",
            "No keyword alignment analysis available."
        )
    )

    # --------------------------------------------------------
    # Section feedback
    # --------------------------------------------------------

    feedback = result.get(
        "section_feedback",
        {}
    )

    if not isinstance(
        feedback,
        dict
    ):

        feedback = {}

    result["section_feedback"] = {
        "summary": str(
            feedback.get(
                "summary",
                "No feedback available."
            )
        ),
        "skills": str(
            feedback.get(
                "skills",
                "No feedback available."
            )
        ),
        "experience": str(
            feedback.get(
                "experience",
                "No feedback available."
            )
        ),
        "education": str(
            feedback.get(
                "education",
                "No feedback available."
            )
        )
    }

    return result


# ============================================================
# ANALYZE RESUME WITH GROQ
# ============================================================

def analyze_resume(
    resume_text,
    job_description
):

    api_key = get_groq_api_key()

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add GROQ_API_KEY to Streamlit Secrets."
        )

    client = Groq(
        api_key=api_key
    )

    if job_description and job_description.strip():

        job_text = job_description.strip()

    else:

        job_text = (
            "No job description was provided. "
            "Analyze the resume for general ATS readiness."
        )

    user_prompt = f"""
Analyze the following resume.

================ RESUME ================

{resume_text}

================ JOB DESCRIPTION ================

{job_text}

================ INSTRUCTIONS ================

Return ONLY a single valid JSON object.

Do not use Markdown.

Do not use code fences.

Do not write anything outside the JSON object.

The top-level JSON object must contain:

1. ats_score
2. score_label
3. summary
4. category_scores
5. strengths
6. improvements
7. missing_keywords
8. formatting_risks
9. keyword_alignment
10. section_feedback

category_scores must contain:

structure_formatting
contact_summary
skills_keywords
experience
education_certifications
clarity_consistency
ats_risk_factors

Use these maximum scores:

structure_formatting = 20
contact_summary = 10
skills_keywords = 20
experience = 25
education_certifications = 10
clarity_consistency = 10
ats_risk_factors = 5

The seven category scores must add up exactly to ats_score.

strengths must be an array of strings.

missing_keywords must be an array of strings.

formatting_risks must be an array of strings.

improvements must be an array of objects.

Each improvement object must contain:

priority
issue
recommendation

section_feedback must contain:

summary
skills
experience
education

Be realistic.

Never invent candidate information.

If the job description mentions a skill that is not supported by
the resume, list it as a potentially missing keyword rather than
claiming the candidate has that skill.

Do not recommend adding unsupported skills.

Keep the response concise.
"""

    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0,

            response_format={
                "type": "json_object"
            }
        )

    except Exception as exc:

        error_text = str(
            exc
        ).lower()

        # Rate limit
        if (
            "429" in error_text
            or "rate limit" in error_text
            or "rate_limit" in error_text
        ):

            raise RuntimeError(
                "Groq rate limit reached. "
                "Please wait and try again."
            ) from exc

        # Authentication
        if (
            "401" in error_text
            or "authentication" in error_text
            or "invalid api key" in error_text
        ):

            raise RuntimeError(
                "Groq API key is invalid. "
                "Check GROQ_API_KEY in Streamlit Secrets."
            ) from exc

        # Model error
        if (
            "model" in error_text
            and (
                "not found" in error_text
                or "does not exist" in error_text
                or "decommissioned" in error_text
            )
        ):

            raise RuntimeError(
                f"The Groq model '{MODEL}' is unavailable."
            ) from exc

        # Generic error
        raise RuntimeError(
            f"Groq API error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Check response
    # --------------------------------------------------------

    if not response.choices:

        raise RuntimeError(
            "Groq returned no response choices."
        )

    message = response.choices[0].message

    output_text = message.content

    if not output_text:

        raise RuntimeError(
            "Groq returned an empty response."
        )

    output_text = output_text.strip()

    # --------------------------------------------------------
    # Remove accidental Markdown fences
    # --------------------------------------------------------

    if output_text.startswith(
        "```json"
    ):

        output_text = output_text[
            len("```json"):
        ]

    elif output_text.startswith(
        "```"
    ):

        output_text = output_text[
            len("```"):
        ]

    if output_text.endswith(
        "```"
    ):

        output_text = output_text[
            :-3
        ]

    output_text = output_text.strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        result = json.loads(
            output_text
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Groq returned invalid JSON.\n\n"
            f"Response received:\n"
            f"{output_text[:1000]}"
        ) from exc

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    return normalize_result(
        result
    )


# ============================================================
# DISPLAY CATEGORY SCORES
# ============================================================

def display_category_scores(
    category_scores
):

    st.subheader(
        "📊 Category Scores"
    )

    categories = [

        (
            "Structure & Formatting",
            "structure_formatting",
            20
        ),

        (
            "Contact & Summary",
            "contact_summary",
            10
        ),

        (
            "Skills & Keywords",
            "skills_keywords",
            20
        ),

        (
            "Experience",
            "experience",
            25
        ),

        (
            "Education & Certifications",
            "education_certifications",
            10
        ),

        (
            "Clarity & Consistency",
            "clarity_consistency",
            10
        ),

        (
            "ATS Risk Factors",
            "ats_risk_factors",
            5
        )
    ]

    cols = st.columns(4)

    for index, (
        label,
        key,
        maximum
    ) in enumerate(categories):

        value = int(
            category_scores.get(
                key,
                0
            )
        )

        with cols[
            index % 4
        ]:

            st.metric(
                label,
                f"{value}/{maximum}"
            )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(result):

    score = int(
        result.get(
            "ats_score",
            0
        )
    )

    label = result.get(
        "score_label",
        get_score_label(score)
    )

    # --------------------------------------------------------
    # Overall score
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🎯 Overall ATS Score"
    )

    score_col, summary_col = st.columns(
        [1, 2]
    )

    with score_col:

        st.metric(
            "ATS Score",
            f"{score}/100"
        )

        st.progress(
            score / 100
        )

        st.caption(
            label
        )

    with summary_col:

        st.markdown(
            "### Resume Summary"
        )

        st.write(
            result.get(
                "summary",
                "No summary available."
            )
        )

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    display_category_scores(
        result.get(
            "category_scores",
            {}
        )
    )

    # --------------------------------------------------------
    # Strengths
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "✅ Strengths"
    )

    strengths = result.get(
        "strengths",
        []
    )

    if strengths:

        for strength in strengths:

            st.success(
                strength
            )

    else:

        st.info(
            "No specific strengths were identified."
        )

    # --------------------------------------------------------
    # Missing keywords
    # --------------------------------------------------------

    st.subheader(
        "🔑 Missing Keywords"
    )

    missing_keywords = result.get(
        "missing_keywords",
        []
    )

    if missing_keywords:

        st.warning(
            ", ".join(
                missing_keywords
            )
        )

    else:

        st.success(
            "No major missing keywords were identified."
        )

    # --------------------------------------------------------
    # Keyword alignment
    # --------------------------------------------------------

    st.subheader(
        "🔎 Keyword Alignment"
    )

    st.write(
        result.get(
            "keyword_alignment",
            "No keyword alignment analysis available."
        )
    )

    # --------------------------------------------------------
    # ATS risks
    # --------------------------------------------------------

    st.subheader(
        "⚠️ Formatting & ATS Risks"
    )

    formatting_risks = result.get(
        "formatting_risks",
        []
    )

    if formatting_risks:

        for risk in formatting_risks:

            st.warning(
                risk
            )

    else:

        st.success(
            "No significant ATS formatting risks were identified."
        )

    # --------------------------------------------------------
    # Improvements
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🛠️ Recommended Improvements"
    )

    improvements = result.get(
        "improvements",
        []
    )

    if improvements:

        for improvement in improvements:

            priority = improvement.get(
                "priority",
                "Medium"
            )

            issue = improvement.get(
                "issue",
                ""
            )

            recommendation = improvement.get(
                "recommendation",
                ""
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

    st.subheader(
        "📝 Section-by-Section Feedback"
    )

    feedback = result.get(
        "section_feedback",
        {}
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Summary",
            "Skills",
            "Experience",
            "Education"
        ]
    )

    with tab1:

        st.write(
            feedback.get(
                "summary",
                "No feedback available."
            )
        )

    with tab2:

        st.write(
            feedback.get(
                "skills",
                "No feedback available."
            )
        )

    with tab3:

        st.write(
            feedback.get(
                "experience",
                "No feedback available."
            )
        )

    with tab4:

        st.write(
            feedback.get(
                "education",
                "No feedback available."
            )
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "📄 Resume ATS Analyzer"
    )

    st.markdown(
        """
### About

Analyze your resume for:

- ATS compatibility
- Resume structure
- Keywords
- Skills
- Experience
- Education
- Certifications
- Formatting risks
- Job-description alignment
"""
    )

    st.divider()

    st.markdown(
        "### 🤖 AI Model"
    )

    st.code(
        MODEL,
        language="text"
    )

    st.caption(
        "Powered by Groq"
    )

    st.divider()

    st.markdown(
        "### 🔒 Privacy"
    )

    st.caption(
        "The application does not store uploaded resumes "
        "in its own database."
    )


# ============================================================
# MAIN UI
# ============================================================

st.title(
    "📄 Resume ATS Analyzer"
)

st.markdown(
    """
Upload your resume and get an ATS-focused analysis.

For the best job-specific results, paste the job description
you are applying for.
"""
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=[
        "pdf",
        "docx",
        "txt"
    ],
    help=(
        "Supported formats: PDF, DOCX and TXT"
    )
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

job_description = st.text_area(
    "Job Description (Optional)",
    height=220,
    placeholder=(
        "Paste the complete job description here "
        "for keyword and job-match analysis..."
    )
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True
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

        # ----------------------------------------------------
        # Extract resume text
        # ----------------------------------------------------

        with st.spinner(
            "Reading your resume..."
        ):

            resume_text = extract_resume_text(
                uploaded_file
            )

            resume_text = clean_text(
                resume_text
            )

        # ----------------------------------------------------
        # Check text
        # ----------------------------------------------------

        if not resume_text:

            st.error(
                "No readable text was found in the uploaded file."
            )

            st.info(
                "If this is a scanned/image-only PDF, "
                "OCR will be required."
            )

            st.stop()

        # ----------------------------------------------------
        # Limit huge resumes
        # ----------------------------------------------------

        MAX_CHARS = 60000

        if len(resume_text) > MAX_CHARS:

            resume_text = resume_text[
                :MAX_CHARS
            ]

            st.warning(
                "The resume is very large. "
                "Only the first 60,000 characters "
                "will be analyzed."
            )

        # ----------------------------------------------------
        # Analyze
        # ----------------------------------------------------

        with st.spinner(
            "Analyzing your resume with Groq AI..."
        ):

            result = analyze_resume(
                resume_text,
                job_description
            )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        st.session_state[
            "ats_result"
        ] = result

        st.success(
            "Resume analysis completed successfully!"
        )

    except Exception as exc:

        st.error(
            f"Analysis failed: {exc}"
        )

        error_text = str(
            exc
        ).lower()

        if "rate limit" in error_text:

            st.warning(
                "Groq's rate limit has been reached. "
                "Please wait and try again later."
            )

        elif (
            "api key" in error_text
            or "authentication" in error_text
        ):

            st.info(
                "Check your GROQ_API_KEY in "
                "Streamlit Cloud → Settings → Secrets."
            )

        elif "json" in error_text:

            st.warning(
                "The AI response could not be processed. "
                "Please try Analyze again."
            )


# ============================================================
# SHOW PREVIOUS RESULT
# ============================================================

if (
    "ats_result"
    in st.session_state
):

    display_results(
        st.session_state[
            "ats_result"
        ]
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Resume ATS Analyzer • AI-assisted resume evaluation • "
    "Always verify recommendations against actual job requirements."
)
