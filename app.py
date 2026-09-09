import base64
import io
import json
import os
import re
import time

import streamlit as st
from google import genai
from docx import Document


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gemini-3.6-flash"
MAX_TEXT_CHARS = 120_000
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_RETRIES = 4


st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📄 Resume ATS Analyzer")

st.caption(
    "Upload a resume to get an AI-estimated ATS compatibility "
    "score and actionable improvements."
)


# ============================================================
# API KEY
# ============================================================

def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None

    return key or os.getenv("GEMINI_API_KEY")


# ============================================================
# DOCX TEXT EXTRACTION
# ============================================================

def extract_docx_text(data: bytes) -> str:

    doc = Document(io.BytesIO(data))

    parts = [
        paragraph.text.strip()
        for paragraph in doc.paragraphs
        if paragraph.text.strip()
    ]

    for table in doc.tables:

        for row in table.rows:

            cells = [
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            ]

            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts)


# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_resume(uploaded_file):

    data = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):

        text = data.decode(
            "utf-8",
            errors="ignore",
        )

        return text, None

    if filename.endswith(".docx"):

        text = extract_docx_text(data)

        return text, None

    if filename.endswith(".pdf"):

        if len(data) > MAX_PDF_BYTES:

            raise ValueError(
                "PDF is larger than 50 MB. "
                "Please upload a smaller PDF."
            )

        return None, data

    raise ValueError(
        "Unsupported file type. "
        "Please upload PDF, DOCX, or TXT."
    )


# ============================================================
# JSON CLEANING
# ============================================================

def clean_json(text: str):

    text = text.strip()

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    return json.loads(text)


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
            "enum": [
                "Poor",
                "Needs Improvement",
                "Good",
                "Very Good",
                "Excellent",
            ],
        },

        "summary": {
            "type": "string",
        },

        "category_scores": {

            "type": "object",

            "properties": {

                "structure_formatting": {
                    "type": "integer",
                },

                "contact_summary": {
                    "type": "integer",
                },

                "skills_keywords": {
                    "type": "integer",
                },

                "experience": {
                    "type": "integer",
                },

                "education_certifications": {
                    "type": "integer",
                },

                "clarity_consistency": {
                    "type": "integer",
                },

                "ats_risk_factors": {
                    "type": "integer",
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
                        "enum": [
                            "High",
                            "Medium",
                            "Low",
                        ],
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
}


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    resume_text,
    job_description,
):

    if resume_text:

        resume_content = resume_text[
            :MAX_TEXT_CHARS
        ]

    else:

        resume_content = (
            "The resume is supplied as a PDF document. "
            "Analyze the PDF document itself."
        )

    if job_description and job_description.strip():

        job_content = job_description.strip()

    else:

        job_content = (
            "No job description provided."
        )

    return f"""

You are an expert ATS resume evaluator
and senior recruiter.

Analyze the uploaded resume conservatively.

IMPORTANT:

- Produce an ATS COMPATIBILITY ESTIMATE.
- Do not claim that the score represents any
  specific company's proprietary ATS.
- Do not invent facts.
- Do not invent experience.
- Do not invent skills.
- Do not invent employers.
- Do not invent dates.
- Do not invent certifications.
- Do not invent achievements.
- Base recommendations only on information
  actually present in the resume.
- If a job description is provided, evaluate
  keyword alignment against it.
- If no job description is provided, evaluate
  general ATS readiness only.

SCORING RUBRIC — 100 POINTS TOTAL:

1. ATS-friendly structure and formatting: 20
2. Contact information and professional summary: 10
3. Skills and keyword coverage: 20
4. Work experience quality, relevance,
   and measurable achievements: 25
5. Education/certifications: 10
6. Clarity, consistency, grammar,
   and readability: 10
7. ATS risk factors: 5

CATEGORY MAXIMUMS:

structure_formatting = 20
contact_summary = 10
skills_keywords = 20
experience = 25
education_certifications = 10
clarity_consistency = 10
ats_risk_factors = 5

Score conservatively if information is missing.

JOB DESCRIPTION:

{job_content}


RESUME:

{resume_content}


Return ONLY valid JSON.

The JSON must contain:

{{
  "ats_score": 0,
  "score_label": "Poor|Needs Improvement|Good|Very Good|Excellent",

  "summary": "short overall assessment",

  "category_scores": {{
    "structure_formatting": 0,
    "contact_summary": 0,
    "skills_keywords": 0,
    "experience": 0,
    "education_certifications": 0,
    "clarity_consistency": 0,
    "ats_risk_factors": 0
  }},

  "strengths": [
    "..."
  ],

  "improvements": [
    {{
      "priority": "High|Medium|Low",
      "issue": "...",
      "recommendation": "..."
    }}
  ],

  "missing_keywords": [
    "..."
  ],

  "formatting_risks": [
    "..."
  ],

  "keyword_alignment":
    "Not evaluated without a job description, or a concise assessment",

  "section_feedback": {{
    "summary": "...",
    "skills": "...",
    "experience": "...",
    "education": "..."
  }}
}}

"""


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_resume(
    resume_text,
    pdf_data,
    job_description,
):

    api_key = get_api_key()

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured. "
            "Add GEMINI_API_KEY to Streamlit "
            "Cloud → Settings → Secrets."
        )

    client = genai.Client(
        api_key=api_key
    )

    prompt = build_prompt(
        resume_text,
        job_description,
    )

    # --------------------------------------------------------
    # Prepare Interactions API input
    # --------------------------------------------------------

    if pdf_data is not None:

        pdf_base64 = base64.b64encode(
            pdf_data
        ).decode("utf-8")

        input_data = [

            {
                "type": "text",
                "text": prompt,
            },

            {
                "type": "document",
                "data": pdf_base64,
                "mime_type": "application/pdf",
            },
        ]

    else:

        input_data = prompt

    last_error = None

    # --------------------------------------------------------
    # Retry temporary Gemini errors
    # --------------------------------------------------------

    for attempt in range(MAX_RETRIES):

        try:

            interaction = client.interactions.create(

                model=MODEL,

                input=input_data,

                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": ATS_SCHEMA,
                },
            )

            output_text = getattr(
                interaction,
                "output_text",
                None,
            )

            if not output_text:

                raise RuntimeError(
                    "Gemini completed the request "
                    "but returned an empty response."
                )

            return clean_json(
                output_text
            )

        import base64
import io
import json
import os
import re
import time

import streamlit as st
from google import genai
from docx import Document


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gemini-3.6-flash"
MAX_TEXT_CHARS = 120_000
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_RETRIES = 4


st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📄 Resume ATS Analyzer")

st.caption(
    "Upload a resume to get an AI-estimated ATS compatibility "
    "score and actionable improvements."
)


# ============================================================
# API KEY
# ============================================================

def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None

    return key or os.getenv("GEMINI_API_KEY")


# ============================================================
# DOCX TEXT EXTRACTION
# ============================================================

def extract_docx_text(data: bytes) -> str:

    doc = Document(io.BytesIO(data))

    parts = [
        paragraph.text.strip()
        for paragraph in doc.paragraphs
        if paragraph.text.strip()
    ]

    for table in doc.tables:

        for row in table.rows:

            cells = [
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            ]

            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts)


# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_resume(uploaded_file):

    data = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):

        text = data.decode(
            "utf-8",
            errors="ignore",
        )

        return text, None

    if filename.endswith(".docx"):

        text = extract_docx_text(data)

        return text, None

    if filename.endswith(".pdf"):

        if len(data) > MAX_PDF_BYTES:

            raise ValueError(
                "PDF is larger than 50 MB. "
                "Please upload a smaller PDF."
            )

        return None, data

    raise ValueError(
        "Unsupported file type. "
        "Please upload PDF, DOCX, or TXT."
    )


# ============================================================
# JSON CLEANING
# ============================================================

def clean_json(text: str):

    text = text.strip()

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    return json.loads(text)


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
            "enum": [
                "Poor",
                "Needs Improvement",
                "Good",
                "Very Good",
                "Excellent",
            ],
        },

        "summary": {
            "type": "string",
        },

        "category_scores": {

            "type": "object",

            "properties": {

                "structure_formatting": {
                    "type": "integer",
                },

                "contact_summary": {
                    "type": "integer",
                },

                "skills_keywords": {
                    "type": "integer",
                },

                "experience": {
                    "type": "integer",
                },

                "education_certifications": {
                    "type": "integer",
                },

                "clarity_consistency": {
                    "type": "integer",
                },

                "ats_risk_factors": {
                    "type": "integer",
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
                        "enum": [
                            "High",
                            "Medium",
                            "Low",
                        ],
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
}


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    resume_text,
    job_description,
):

    if resume_text:

        resume_content = resume_text[
            :MAX_TEXT_CHARS
        ]

    else:

        resume_content = (
            "The resume is supplied as a PDF document. "
            "Analyze the PDF document itself."
        )

    if job_description and job_description.strip():

        job_content = job_description.strip()

    else:

        job_content = (
            "No job description provided."
        )

    return f"""

You are an expert ATS resume evaluator
and senior recruiter.

Analyze the uploaded resume conservatively.

IMPORTANT:

- Produce an ATS COMPATIBILITY ESTIMATE.
- Do not claim that the score represents any
  specific company's proprietary ATS.
- Do not invent facts.
- Do not invent experience.
- Do not invent skills.
- Do not invent employers.
- Do not invent dates.
- Do not invent certifications.
- Do not invent achievements.
- Base recommendations only on information
  actually present in the resume.
- If a job description is provided, evaluate
  keyword alignment against it.
- If no job description is provided, evaluate
  general ATS readiness only.

SCORING RUBRIC — 100 POINTS TOTAL:

1. ATS-friendly structure and formatting: 20
2. Contact information and professional summary: 10
3. Skills and keyword coverage: 20
4. Work experience quality, relevance,
   and measurable achievements: 25
5. Education/certifications: 10
6. Clarity, consistency, grammar,
   and readability: 10
7. ATS risk factors: 5

CATEGORY MAXIMUMS:

structure_formatting = 20
contact_summary = 10
skills_keywords = 20
experience = 25
education_certifications = 10
clarity_consistency = 10
ats_risk_factors = 5

Score conservatively if information is missing.

JOB DESCRIPTION:

{job_content}


RESUME:

{resume_content}


Return ONLY valid JSON.

The JSON must contain:

{{
  "ats_score": 0,
  "score_label": "Poor|Needs Improvement|Good|Very Good|Excellent",

  "summary": "short overall assessment",

  "category_scores": {{
    "structure_formatting": 0,
    "contact_summary": 0,
    "skills_keywords": 0,
    "experience": 0,
    "education_certifications": 0,
    "clarity_consistency": 0,
    "ats_risk_factors": 0
  }},

  "strengths": [
    "..."
  ],

  "improvements": [
    {{
      "priority": "High|Medium|Low",
      "issue": "...",
      "recommendation": "..."
    }}
  ],

  "missing_keywords": [
    "..."
  ],

  "formatting_risks": [
    "..."
  ],

  "keyword_alignment":
    "Not evaluated without a job description, or a concise assessment",

  "section_feedback": {{
    "summary": "...",
    "skills": "...",
    "experience": "...",
    "education": "..."
  }}
}}

"""


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_resume(
    resume_text,
    pdf_data,
    job_description,
):

    api_key = get_api_key()

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured. "
            "Add GEMINI_API_KEY to Streamlit "
            "Cloud → Settings → Secrets."
        )

    client = genai.Client(
        api_key=api_key
    )

    prompt = build_prompt(
        resume_text,
        job_description,
    )

    # --------------------------------------------------------
    # Prepare Interactions API input
    # --------------------------------------------------------

    if pdf_data is not None:

        pdf_base64 = base64.b64encode(
            pdf_data
        ).decode("utf-8")

        input_data = [

            {
                "type": "text",
                "text": prompt,
            },

            {
                "type": "document",
                "data": pdf_base64,
                "mime_type": "application/pdf",
            },
        ]

    else:

        input_data = prompt

    last_error = None

    # --------------------------------------------------------
    # Retry temporary Gemini errors
    # --------------------------------------------------------

    for attempt in range(MAX_RETRIES):

        try:

         interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=input_data,
    response_format={
        "type": "text",
        "mime_type": "application/json",
        "schema": ATS_SCHEMA,
    },
)

output_text = interaction.output_text

if not output_text:
    raise RuntimeError(
        "Gemini returned an empty response."
    )

return clean_json(output_text)

        except Exception as exc:

            last_error = exc

            error_text = str(exc).upper()

            # ------------------------------------------------
            # Permanent model/API errors
            # ------------------------------------------------

            if (
                "404" in error_text
                or "NOT_FOUND" in error_text
            ):

                raise RuntimeError(
                    f"Gemini model '{MODEL}' was not found "
                    "or is not available to this API key. "
                    "Check that your Google AI API key has "
                    "access to Gemini 3.6 Flash."
                ) from exc

            if (
                "400" in error_text
                or "INVALID_ARGUMENT" in error_text
            ):

                raise RuntimeError(
                    f"Gemini rejected the request: {exc}"
                ) from exc

            # ------------------------------------------------
            # Temporary errors
            # ------------------------------------------------

            temporary_error = any(
                code in error_text
                for code in [
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "500",
                    "INTERNAL",
                    "503",
                    "UNAVAILABLE",
                    "504",
                    "DEADLINE_EXCEEDED",
                ]
            )

            if not temporary_error:

                raise

            if attempt < MAX_RETRIES - 1:

                wait_seconds = 2 ** (
                    attempt + 1
                )

                st.warning(
                    "Gemini is temporarily busy "
                    "or unavailable. "
                    f"Retrying in {wait_seconds} seconds "
                    f"({attempt + 1}/{MAX_RETRIES - 1})..."
                )

                time.sleep(
                    wait_seconds
                )

    raise RuntimeError(
        "Gemini is temporarily unavailable "
        "after multiple attempts. "
        "Please wait a little and try again."
    ) from last_error


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Settings")

    st.info(
        "Add your Gemini API key in "
        "Streamlit Cloud → App settings → Secrets."
    )

    st.markdown(
        f"**Model:** `{MODEL}`"
    )

    st.markdown(
        "**API:** Gemini Interactions API"
    )

    st.divider()

    st.caption(
        "PDF, DOCX and TXT resumes are supported."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload your resume",
    type=[
        "pdf",
        "docx",
        "txt",
    ],
    help=(
        "PDF, DOCX, and TXT files "
        "are supported."
    ),
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

job_description = st.text_area(
    "Optional: paste the job description",
    height=180,
    placeholder=(
        "Adding the target job description "
        "makes keyword matching and the ATS "
        "estimate more useful."
    ),
)


if uploaded:

    st.success(
        f"Loaded: {uploaded.name}"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Resume",
    type="primary",
    disabled=uploaded is None,
):

    try:

        with st.spinner(
            "Analyzing resume with Gemini 3.6 Flash..."
        ):

            resume_text, pdf_data = (
                extract_resume(uploaded)
            )

            if (
                resume_text is not None
                and not resume_text.strip()
            ):

                st.error(
                    "No readable text was found "
                    "in the uploaded file."
                )

                st.stop()

            result = analyze_resume(
                resume_text,
                pdf_data,
                job_description,
            )

        # ====================================================
        # ATS SCORE
        # ====================================================

        score = int(
            result.get(
                "ats_score",
                0,
            )
        )

        score = max(
            0,
            min(100, score),
        )

        label = result.get(
            "score_label",
            "Needs Improvement",
        )

        st.subheader(
            "ATS Compatibility Estimate"
        )

        c1, c2 = st.columns(
            [1, 3]
        )

        with c1:

            st.metric(
                "Score",
                f"{score}/100",
            )

        with c2:

            st.progress(
                score / 100
            )

            st.write(
                f"**{label}**"
            )

            st.write(
                result.get(
                    "summary",
                    "",
                )
            )

        # ====================================================
        # CATEGORY SCORES
        # ====================================================

        st.subheader(
            "Category Scores"
        )

        scores = result.get(
            "category_scores",
            {},
        )

        categories = [

            (
                "Structure & formatting",
                "structure_formatting",
                20,
            ),

            (
                "Contact & summary",
                "contact_summary",
                10,
            ),

            (
                "Skills & keywords",
                "skills_keywords",
                20,
            ),

            (
                "Experience",
                "experience",
                25,
            ),

            (
                "Education & certifications",
                "education_certifications",
                10,
            ),

            (
                "Clarity & consistency",
                "clarity_consistency",
                10,
            ),

            (
                "ATS risk factors",
                "ats_risk_factors",
                5,
            ),
        ]

        cols = st.columns(4)

        for i, (
            label_text,
            key,
            maximum,
        ) in enumerate(categories):

            with cols[i % 4]:

                try:

                    value = int(
                        scores.get(
                            key,
                            0,
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    value = 0

                value = max(
                    0,
                    min(
                        maximum,
                        value,
                    ),
                )

                st.metric(
                    label_text,
                    f"{value}/{maximum}",
                )

        # ====================================================
        # STRENGTHS AND RISKS
        # ====================================================

        left, right = st.columns(2)

        with left:

            st.subheader(
                "✅ Strengths"
            )

            strengths = result.get(
                "strengths",
                [],
            )

            if strengths:

                for item in strengths:

                    st.markdown(
                        f"- {item}"
                    )

            else:

                st.write(
                    "No strengths were returned."
                )

            st.subheader(
                "🔑 Missing / Weak Keywords"
            )

            keywords = result.get(
                "missing_keywords",
                [],
            )

            if keywords:

                st.write(
                    ", ".join(keywords)
                )

            else:

                st.write(
                    "No major missing keywords identified."
                )

        with right:

            st.subheader(
                "⚠️ Formatting Risks"
            )

            risks = result.get(
                "formatting_risks",
                [],
            )

            if risks:

                for item in risks:

                    st.markdown(
                        f"- {item}"
                    )

            else:

                st.write(
                    "No major ATS formatting risks identified."
                )

        # ====================================================
        # IMPROVEMENTS
        # ====================================================

        st.subheader(
            "🛠️ Recommended Improvements"
        )

        improvements = result.get(
            "improvements",
            [],
        )

        if improvements:

            for item in improvements:

                priority = item.get(
                    "priority",
                    "Medium",
                )

                icon = {
                    "High": "🔴",
                    "Medium": "🟠",
                    "Low": "🟢",
                }.get(
                    priority,
                    "🔵",
                )

                with st.expander(
                    f"{icon} {priority}: "
                    f"{item.get('issue', 'Improvement')}"
                ):

                    st.write(
                        item.get(
                            "recommendation",
                            "",
                        )
                    )

        else:

            st.write(
                "No specific improvements were returned."
            )

        # ====================================================
        # SECTION FEEDBACK
        # ====================================================

        st.subheader(
            "📌 Section Feedback"
        )

        feedback = result.get(
            "section_feedback",
            {},
        )

        for section in [
            "summary",
            "skills",
            "experience",
            "education",
        ]:

            st.markdown(
                f"**{section.title()}**"
            )

            st.write(
                feedback.get(
                    section,
                    "No feedback provided.",
                )
            )

        # ====================================================
        # JOB MATCH
        # ====================================================

        st.subheader(
            "🎯 Job Match"
        )

        st.write(
            result.get(
                "keyword_alignment",
                "",
            )
        )

        st.caption(
            "Important: this is an AI-based ATS "
            "compatibility estimate. Different ATS "
            "platforms and employers use different "
            "parsing and ranking rules."
        )

    except json.JSONDecodeError:

        st.error(
            "Gemini returned a response that "
            "could not be parsed as JSON. "
            "Please try again."
        )

    except Exception as exc:

        st.error(
            f"Analysis failed: {exc}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Privacy note: resumes contain personal information. "
    "Avoid uploading sensitive documents you do not want "
    "processed by a third-party AI service."
)

            # ------------------------------------------------
            # Permanent model/API errors
            # ------------------------------------------------

            if (
                "404" in error_text
                or "NOT_FOUND" in error_text
            ):

                raise RuntimeError(
                    f"Gemini model '{MODEL}' was not found "
                    "or is not available to this API key. "
                    "Check that your Google AI API key has "
                    "access to Gemini 3.6 Flash."
                ) from exc

            if (
                "400" in error_text
                or "INVALID_ARGUMENT" in error_text
            ):

                raise RuntimeError(
                    f"Gemini rejected the request: {exc}"
                ) from exc

            # ------------------------------------------------
            # Temporary errors
            # ------------------------------------------------

            temporary_error = any(
                code in error_text
                for code in [
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "500",
                    "INTERNAL",
                    "503",
                    "UNAVAILABLE",
                    "504",
                    "DEADLINE_EXCEEDED",
                ]
            )

            if not temporary_error:

                raise

            if attempt < MAX_RETRIES - 1:

                wait_seconds = 2 ** (
                    attempt + 1
                )

                st.warning(
                    "Gemini is temporarily busy "
                    "or unavailable. "
                    f"Retrying in {wait_seconds} seconds "
                    f"({attempt + 1}/{MAX_RETRIES - 1})..."
                )

                time.sleep(
                    wait_seconds
                )

    raise RuntimeError(
        "Gemini is temporarily unavailable "
        "after multiple attempts. "
        "Please wait a little and try again."
    ) from last_error


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Settings")

    st.info(
        "Add your Gemini API key in "
        "Streamlit Cloud → App settings → Secrets."
    )

    st.markdown(
        f"**Model:** `{MODEL}`"
    )

    st.markdown(
        "**API:** Gemini Interactions API"
    )

    st.divider()

    st.caption(
        "PDF, DOCX and TXT resumes are supported."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload your resume",
    type=[
        "pdf",
        "docx",
        "txt",
    ],
    help=(
        "PDF, DOCX, and TXT files "
        "are supported."
    ),
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

job_description = st.text_area(
    "Optional: paste the job description",
    height=180,
    placeholder=(
        "Adding the target job description "
        "makes keyword matching and the ATS "
        "estimate more useful."
    ),
)


if uploaded:

    st.success(
        f"Loaded: {uploaded.name}"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Resume",
    type="primary",
    disabled=uploaded is None,
):

    try:

        with st.spinner(
            "Analyzing resume with Gemini 3.6 Flash..."
        ):

            resume_text, pdf_data = (
                extract_resume(uploaded)
            )

            if (
                resume_text is not None
                and not resume_text.strip()
            ):

                st.error(
                    "No readable text was found "
                    "in the uploaded file."
                )

                st.stop()

            result = analyze_resume(
                resume_text,
                pdf_data,
                job_description,
            )

        # ====================================================
        # ATS SCORE
        # ====================================================

        score = int(
            result.get(
                "ats_score",
                0,
            )
        )

        score = max(
            0,
            min(100, score),
        )

        label = result.get(
            "score_label",
            "Needs Improvement",
        )

        st.subheader(
            "ATS Compatibility Estimate"
        )

        c1, c2 = st.columns(
            [1, 3]
        )

        with c1:

            st.metric(
                "Score",
                f"{score}/100",
            )

        with c2:

            st.progress(
                score / 100
            )

            st.write(
                f"**{label}**"
            )

            st.write(
                result.get(
                    "summary",
                    "",
                )
            )

        # ====================================================
        # CATEGORY SCORES
        # ====================================================

        st.subheader(
            "Category Scores"
        )

        scores = result.get(
            "category_scores",
            {},
        )

        categories = [

            (
                "Structure & formatting",
                "structure_formatting",
                20,
            ),

            (
                "Contact & summary",
                "contact_summary",
                10,
            ),

            (
                "Skills & keywords",
                "skills_keywords",
                20,
            ),

            (
                "Experience",
                "experience",
                25,
            ),

            (
                "Education & certifications",
                "education_certifications",
                10,
            ),

            (
                "Clarity & consistency",
                "clarity_consistency",
                10,
            ),

            (
                "ATS risk factors",
                "ats_risk_factors",
                5,
            ),
        ]

        cols = st.columns(4)

        for i, (
            label_text,
            key,
            maximum,
        ) in enumerate(categories):

            with cols[i % 4]:

                try:

                    value = int(
                        scores.get(
                            key,
                            0,
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    value = 0

                value = max(
                    0,
                    min(
                        maximum,
                        value,
                    ),
                )

                st.metric(
                    label_text,
                    f"{value}/{maximum}",
                )

        # ====================================================
        # STRENGTHS AND RISKS
        # ====================================================

        left, right = st.columns(2)

        with left:

            st.subheader(
                "✅ Strengths"
            )

            strengths = result.get(
                "strengths",
                [],
            )

            if strengths:

                for item in strengths:

                    st.markdown(
                        f"- {item}"
                    )

            else:

                st.write(
                    "No strengths were returned."
                )

            st.subheader(
                "🔑 Missing / Weak Keywords"
            )

            keywords = result.get(
                "missing_keywords",
                [],
            )

            if keywords:

                st.write(
                    ", ".join(keywords)
                )

            else:

                st.write(
                    "No major missing keywords identified."
                )

        with right:

            st.subheader(
                "⚠️ Formatting Risks"
            )

            risks = result.get(
                "formatting_risks",
                [],
            )

            if risks:

                for item in risks:

                    st.markdown(
                        f"- {item}"
                    )

            else:

                st.write(
                    "No major ATS formatting risks identified."
                )

        # ====================================================
        # IMPROVEMENTS
        # ====================================================

        st.subheader(
            "🛠️ Recommended Improvements"
        )

        improvements = result.get(
            "improvements",
            [],
        )

        if improvements:

            for item in improvements:

                priority = item.get(
                    "priority",
                    "Medium",
                )

                icon = {
                    "High": "🔴",
                    "Medium": "🟠",
                    "Low": "🟢",
                }.get(
                    priority,
                    "🔵",
                )

                with st.expander(
                    f"{icon} {priority}: "
                    f"{item.get('issue', 'Improvement')}"
                ):

                    st.write(
                        item.get(
                            "recommendation",
                            "",
                        )
                    )

        else:

            st.write(
                "No specific improvements were returned."
            )

        # ====================================================
        # SECTION FEEDBACK
        # ====================================================

        st.subheader(
            "📌 Section Feedback"
        )

        feedback = result.get(
            "section_feedback",
            {},
        )

        for section in [
            "summary",
            "skills",
            "experience",
            "education",
        ]:

            st.markdown(
                f"**{section.title()}**"
            )

            st.write(
                feedback.get(
                    section,
                    "No feedback provided.",
                )
            )

        # ====================================================
        # JOB MATCH
        # ====================================================

        st.subheader(
            "🎯 Job Match"
        )

        st.write(
            result.get(
                "keyword_alignment",
                "",
            )
        )

        st.caption(
            "Important: this is an AI-based ATS "
            "compatibility estimate. Different ATS "
            "platforms and employers use different "
            "parsing and ranking rules."
        )

    except json.JSONDecodeError:

        st.error(
            "Gemini returned a response that "
            "could not be parsed as JSON. "
            "Please try again."
        )

    except Exception as exc:

        st.error(
            f"Analysis failed: {exc}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Privacy note: resumes contain personal information. "
    "Avoid uploading sensitive documents you do not want "
    "processed by a third-party AI service."
)
