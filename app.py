"""
Resume Review Agent
--------------------
A beginner-friendly, single-agent app built with CrewAI + Streamlit + Groq.

The agent takes:
  1. A candidate's resume (pasted text OR uploaded PDF)
  2. A target job description (pasted text), with optional job title/URL

...and produces a LinkedIn "Job Match"-style qualification checklist
(✓ matched / ? unmatched with reasons) plus a structured, actionable
evaluation of how well the resume matches the job — without inventing
skills or experience the candidate doesn't actually have.
"""

import os
import streamlit as st
from pypdf import PdfReader
from crewai import Agent, Task, Crew, Process
from crewai import LLM


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Resume Review Agent", page_icon="📄", layout="centered")
st.title("📄 Resume Review Agent")
st.write(
    "Paste or upload a resume, paste a target job description, and get a "
    "structured, honest evaluation of the match — plus concrete suggestions "
    "to improve it."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_pdf_text(uploaded_file) -> str:
    """Extract text from an uploaded PDF file. Returns '' on failure."""
    try:
        reader = PdfReader(uploaded_file)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        text = "\n".join(text_parts).strip()
        return text
    except Exception as e:
        st.error(
            "⚠️ I couldn't read that PDF. It may be scanned/image-based, "
            "password-protected, or corrupted. Try pasting the resume text "
            f"instead.\n\nDetails: {e}"
        )
        return ""


def get_groq_api_key() -> str:
    """Fetch the Groq API key from Streamlit secrets, with a clear error if missing."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error(
            "⚠️ No Groq API key found. Add it to `.streamlit/secrets.toml` "
            "locally, or in your app's *Settings → Secrets* on Streamlit "
            "Community Cloud, as:\n\n```toml\nGROQ_API_KEY = \"your-key-here\"\n```"
        )
        st.stop()


def build_crew(resume_text: str, job_description: str, job_title: str = "", job_url: str = "") -> Crew:
    """Construct the single-agent CrewAI crew for this run."""
    groq_api_key = get_groq_api_key()

    # Groq exposes an OpenAI-compatible endpoint, so we use CrewAI's native
    # "openai" provider pointed at Groq's base URL instead of relying on
    # LiteLLM (which isn't installed by default in CrewAI 1.x).
    llm = LLM(
        model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
        provider="openai",
        temperature=0.3,
    )

    reviewer = Agent(
        role="Senior Technical Recruiter",
        goal=(
            "Evaluate how well a candidate's resume matches a target job "
            "description, being accurate and never inventing skills or "
            "experience that aren't actually present in the resume."
        ),
        backstory=(
            "You are a meticulous, experienced recruiter who has screened "
            "thousands of resumes. You give direct, honest, and constructive "
            "feedback. You never fabricate qualifications the candidate does "
            "not have, and you always ground your feedback strictly in the "
            "resume text and job description provided to you."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    job_title_line = job_title.strip() if job_title and job_title.strip() else "this role"
    job_url_line = job_url.strip() if job_url and job_url.strip() else ""

    task = Task(
        description=(
            "You will be given a RESUME and a JOB DESCRIPTION below, and "
            f"optionally a JOB TITLE ({job_title_line}) and JOB URL "
            f"({job_url_line or 'not provided'}).\n\n"
            "RESUME:\n"
            "----------------\n"
            f"{resume_text}\n"
            "----------------\n\n"
            "JOB DESCRIPTION:\n"
            "----------------\n"
            f"{job_description}\n"
            "----------------\n\n"
            "Evaluate the resume against the job description in two parts.\n\n"
            "PART A - Qualification Checklist (LinkedIn 'Job Match' style):\n"
            "1. Read the job description and separate its requirements into "
            "'required qualifications' (must-haves, e.g. years of experience, "
            "specific languages/frameworks/tools, degrees, certifications) "
            "and 'additional qualifications' (nice-to-haves/preferred).\n"
            "2. Separately, identify any qualifications mentioned that are "
            "inherently NOT verifiable from a resume/application (soft skills "
            "like communication, problem-solving, ability to work remotely, "
            "teamwork, culture fit) — these belong in a third list of "
            "qualifications typically evaluated during the application or "
            "interview, not scored as matched/unmatched.\n"
            "3. For every required and additional qualification, check the "
            "resume for clear evidence. Mark it MATCHED only if the resume "
            "explicitly supports it (or very clearly implies it); otherwise "
            "mark it UNMATCHED and give a short, specific reason (e.g. "
            "'No mention of PHP experience', or 'Requires 3-5 years, resume "
            "shows 0-2 years'). Never mark something matched by assuming or "
            "inventing experience not stated in the resume.\n"
            "4. Compute an overall job-match level of Low, Medium, or High "
            "based on the proportion of required qualifications matched "
            "(roughly: 0-40% matched = Low, 41-74% = Medium, 75-100% = High).\n\n"
            "PART B - Detailed Report:\n"
            "Using the same qualification checklist you just built, write a "
            "match score, strengths, gaps, and recommendations as described "
            "in the expected output.\n\n"
            "Base every statement strictly on the text provided — do NOT "
            "assume or invent any skill, tool, or experience that isn't "
            "explicitly stated or clearly implied in the resume. If "
            "information is missing or unclear, say so rather than guessing."
        ),
        expected_output=(
            "A structured Markdown report with EXACTLY these sections, in "
            "this order:\n\n"
            "## Job Match Summary\n"
            "First line, formatted exactly like this (omit the link brackets "
            "and just show the plain title if no URL was provided):\n"
            f"`Job match is {{Low/Medium/High}} - [{job_title_line}]"
            f"({job_url_line or '#'})`\n\n"
            "Then one short sentence summarizing the overall fit, similar in "
            "tone to: 'Your profile and resume are missing some required "
            "qualifications...' or 'Your profile is a strong match for this "
            "role...' as appropriate.\n\n"
            "Then a line: `Matches X of Y required qualifications:` followed "
            "by one bullet per required qualification, each starting with "
            "`✓` if matched (just the qualification text) or `?` if "
            "unmatched (the qualification text, followed by the specific "
            "reason in parentheses).\n\n"
            "Then a line: `Matches X of Y additional qualifications:` "
            "followed by the same ✓ / ? bullet format for each additional "
            "(preferred) qualification. Omit this whole block if the job "
            "description has no additional/preferred qualifications.\n\n"
            "Then, if applicable: `There are qualifications that will "
            "likely be evaluated in the application or interview:` followed "
            "by a `•` bullet for each non-verifiable soft-skill qualification "
            "identified. Omit this block if there are none.\n\n"
            "## Match Score\n"
            "A percentage (0-100%) estimating overall fit, with one sentence "
            "justifying the number.\n\n"
            "## Key Strengths\n"
            "3-5 bullet points on resume elements that align well with the "
            "job description.\n\n"
            "## Gaps & Missing Qualifications\n"
            "3-5 bullet points on requirements from the job description that "
            "are missing, weak, or unclear in the resume. Be specific.\n\n"
            "## Actionable Recommendations\n"
            "3-5 concrete, specific suggestions the candidate can act on to "
            "improve their resume or candidacy for this exact role (e.g. "
            "rewording a bullet, adding a quantifiable result, highlighting "
            "a relevant project, or acquiring a specific skill/certification). "
            "Do not suggest lying or fabricating experience."
        ),
        agent=reviewer,
    )

    return Crew(agents=[reviewer], tasks=[task], process=Process.sequential, verbose=False)


# ---------------------------------------------------------------------------
# UI — Inputs
# ---------------------------------------------------------------------------
st.subheader("1. Resume")
resume_input_mode = st.radio(
    "How would you like to provide the resume?",
    ["Paste text", "Upload PDF"],
    horizontal=True,
)

resume_text = ""
if resume_input_mode == "Paste text":
    resume_text = st.text_area("Paste the resume text here", height=220)
else:
    uploaded_pdf = st.file_uploader("Upload resume (PDF)", type=["pdf"])
    if uploaded_pdf is not None:
        with st.spinner("Extracting text from PDF..."):
            resume_text = extract_pdf_text(uploaded_pdf)
        if resume_text:
            with st.expander("Preview extracted resume text"):
                st.text(resume_text[:3000] + ("..." if len(resume_text) > 3000 else ""))

st.subheader("2. Target Job Description")
col_title, col_url = st.columns(2)
with col_title:
    job_title = st.text_input("Job title (optional)", placeholder="e.g. AI Engineer")
with col_url:
    job_url = st.text_input("Job posting URL (optional)", placeholder="e.g. https://www.linkedin.com/jobs/view/...")
job_description = st.text_area("Paste the job description here", height=220)

st.subheader("3. Run the Review")
run_button = st.button("🔍 Review Resume", type="primary")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run_button:
    # --- Input validation -------------------------------------------------
    if not resume_text or not resume_text.strip():
        st.warning("⚠️ Please provide a resume (paste text or upload a valid PDF) before running.")
        st.stop()

    if not job_description or not job_description.strip():
        st.warning("⚠️ Please paste a target job description before running.")
        st.stop()

    # --- Run the crew, handling Groq/runtime errors gracefully ------------
    try:
        with st.spinner("Analyzing resume against job description... this can take up to a minute."):
            crew = build_crew(resume_text, job_description, job_title, job_url)
            result = crew.kickoff()

        st.success("✅ Review complete!")
        st.markdown("---")
        st.markdown(str(result))

    except Exception as e:
        error_message = str(e).lower()

        if "rate limit" in error_message or "429" in error_message:
            st.error(
                "⚠️ Groq's API rate limit was hit. Please wait a moment and "
                "try again. If this keeps happening, check your Groq usage "
                "tier/limits."
            )
        elif "api key" in error_message or "401" in error_message or "unauthorized" in error_message:
            st.error(
                "⚠️ Groq rejected the API key. Double-check that "
                "`GROQ_API_KEY` in your Streamlit secrets is correct and active."
            )
        elif "timeout" in error_message:
            st.error("⚠️ The request to Groq timed out. Please try again.")
        else:
            st.error(f"⚠️ Something went wrong while generating the review.\n\nDetails: {e}")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Built with CrewAI, Streamlit, and Groq (openai/gpt-oss-120b). "
    "This tool provides guidance only and does not guarantee interview or job outcomes."
)
