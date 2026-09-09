import io
import json
import os
import re

import streamlit as st
from google import genai
from google.genai import types
from docx import Document

MODEL = "gemini-2.5-flash"
MAX_TEXT_CHARS = 120_000

st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume ATS Analyzer")
st.caption("Upload a resume to get an AI-estimated ATS compatibility score and actionable improvements.")


def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("GEMINI_API_KEY")


def extract_docx_text(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def extract_text(uploaded_file):
    data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    mime = uploaded_file.type or "application/octet-stream"

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore"), None
    if name.endswith(".docx"):
        return extract_docx_text(data), None
    if name.endswith(".pdf"):
        # Keep the PDF itself for Gemini because Gemini 2.5 Flash supports PDF/document input.
        return None, types.Part.from_bytes(data=data, mime_type="application/pdf")
    raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


def clean_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def analyze_resume(resume_text, resume_part, job_description):
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured. Add it to Streamlit Secrets or your environment.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an expert ATS resume evaluator and senior recruiter.
Analyze the uploaded resume conservatively. Produce an ATS COMPATIBILITY ESTIMATE, not a claim about any specific company's proprietary ATS.

Scoring rubric (100 points total):
1. ATS-friendly structure and formatting: 20
2. Contact information and professional summary: 10
3. Skills and keyword coverage: 20
4. Work experience quality, relevance, and measurable achievements: 25
5. Education/certifications: 10
6. Clarity, consistency, grammar, and readability: 10
7. ATS risk factors (tables, columns, graphics, headers/footers, unusual symbols, etc.): 5

If a job description is provided, assess keyword alignment against it. If none is provided, score general ATS readiness and say that job-specific keyword matching was not performed.
Do not invent facts, experience, skills, employers, dates, or qualifications.
Give practical improvements that the user can actually make.

Return ONLY valid JSON with this exact top-level structure:
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
  "strengths": ["..."],
  "improvements": [
    {{"priority":"High|Medium|Low", "issue":"...", "recommendation":"..."}}
  ],
  "missing_keywords": ["..."],
  "formatting_risks": ["..."],
  "keyword_alignment": "Not evaluated without a job description, or a concise assessment",
  "section_feedback": {{
    "summary": "...",
    "skills": "...",
    "experience": "...",
    "education": "..."
  }}
}}

JOB DESCRIPTION:
{job_description.strip() if job_description.strip() else "No job description provided."}

RESUME TEXT:
{resume_text[:MAX_TEXT_CHARS] if resume_text else "The resume is supplied as a PDF document attachment. Analyze the document itself."}
"""

    contents = [prompt]
    if resume_part is not None:
        contents.append(resume_part)

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    return clean_json(response.text)


with st.sidebar:
    st.header("Settings")
    st.info("Add your Gemini API key in Streamlit Cloud → App settings → Secrets.")
    st.markdown("**Model:** `gemini-2.5-flash`")

uploaded = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"],
    help="PDF, DOCX, and TXT are supported.",
)

job_description = st.text_area(
    "Optional: paste the job description",
    height=180,
    placeholder="Adding the target job description makes keyword matching and the ATS estimate more useful.",
)

if uploaded:
    st.success(f"Loaded: {uploaded.name}")

if st.button("🔍 Analyze Resume", type="primary", disabled=uploaded is None):
    try:
        with st.spinner("Analyzing resume with Gemini 2.5 Flash..."):
            resume_text, resume_part = extract_text(uploaded)
            if resume_text is not None and not resume_text.strip():
                st.error("No readable text was found in the uploaded file.")
                st.stop()
            result = analyze_resume(resume_text, resume_part, job_description)

        score = int(result.get("ats_score", 0))
        score = max(0, min(100, score))
        label = result.get("score_label", "Needs Improvement")

        st.subheader("ATS Compatibility Estimate")
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Score", f"{score}/100")
        with c2:
            st.progress(score / 100)
            st.write(f"**{label}**")
            st.write(result.get("summary", ""))

        st.subheader("Category Scores")
        scores = result.get("category_scores", {})
        cols = st.columns(4)
        labels = [
            ("Structure & formatting", "structure_formatting"),
            ("Contact & summary", "contact_summary"),
            ("Skills & keywords", "skills_keywords"),
            ("Experience", "experience"),
            ("Education & certifications", "education_certifications"),
            ("Clarity & consistency", "clarity_consistency"),
            ("ATS risk factors", "ats_risk_factors"),
        ]
        for i, (label_text, key) in enumerate(labels):
            with cols[i % 4]:
                st.metric(label_text, f"{scores.get(key, 0)}/" + str({
                    "structure_formatting": 20,
                    "contact_summary": 10,
                    "skills_keywords": 20,
                    "experience": 25,
                    "education_certifications": 10,
                    "clarity_consistency": 10,
                    "ats_risk_factors": 5,
                }[key]))

        left, right = st.columns(2)
        with left:
            st.subheader("✅ Strengths")
            for item in result.get("strengths", []):
                st.markdown(f"- {item}")

            st.subheader("🔑 Missing / Weak Keywords")
            keywords = result.get("missing_keywords", [])
            if keywords:
                st.write(", ".join(keywords))
            else:
                st.write("No major missing keywords identified.")

        with right:
            st.subheader("⚠️ Formatting Risks")
            risks = result.get("formatting_risks", [])
            if risks:
                for item in risks:
                    st.markdown(f"- {item}")
            else:
                st.write("No major ATS formatting risks identified.")

        st.subheader("🛠️ Recommended Improvements")
        improvements = result.get("improvements", [])
        for item in improvements:
            priority = item.get("priority", "Medium")
            icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(priority, "🔵")
            with st.expander(f"{icon} {priority}: {item.get('issue', 'Improvement')}"):
                st.write(item.get("recommendation", ""))

        st.subheader("📌 Section Feedback")
        feedback = result.get("section_feedback", {})
        for section in ["summary", "skills", "experience", "education"]:
            st.markdown(f"**{section.title()}**")
            st.write(feedback.get(section, "No feedback provided."))

        st.subheader("🎯 Job Match")
        st.write(result.get("keyword_alignment", ""))
        st.caption("Important: this is an AI-based ATS compatibility estimate. Different ATS platforms and employers use different parsing and ranking rules.")

    except json.JSONDecodeError:
        st.error("Gemini returned an invalid JSON response. Please try the analysis again.")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

st.divider()
st.caption("Privacy note: resumes contain personal information. Avoid uploading sensitive documents you do not want processed by a third-party AI service.")
