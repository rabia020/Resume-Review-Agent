"""
Resume Review Agent
--------------------
A beginner-friendly, single-agent app built with CrewAI + Streamlit + Groq.

The agent takes:
  1. A candidate resume (pasted text OR uploaded PDF)
  2. A target job title + requirements description

...and returns a structured Dossier Audit style assessment: a match
coefficient, matched competencies, detected gaps, and a strategic
recommendation - without inventing skills or experience the candidate
does not actually have.
"""

import json
import re

import streamlit as st
from pypdf import PdfReader
from crewai import Agent, Task, Crew, Process
from crewai import LLM


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Review Agent", page_icon="📄", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Top bar */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #E5E7EB;
    padding-bottom: 1rem;
    margin-bottom: 2rem;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: 1.05rem;
    color: #0F172A;
}
.topbar-brand .icon-box {
    background: #2F6FED;
    color: white;
    width: 30px;
    height: 30px;
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
}
.topbar-status {
    font-size: 0.8rem;
    color: #64748B;
    font-weight: 500;
}
.topbar-status span {
    color: #16A34A;
    font-weight: 700;
}

/* Eyebrow pill */
.eyebrow-pill {
    display: inline-block;
    background: #EFF6FF;
    color: #2F6FED;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 5px 12px;
    border-radius: 6px;
    margin-bottom: 1.1rem;
}
.eyebrow-plain {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #94A3B8;
    margin-bottom: 0.35rem;
}
.eyebrow-danger {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #DC2626;
    margin-bottom: 0.6rem;
}

/* Hero */
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.08;
    color: #0F172A;
    margin-bottom: 0.9rem;
}
.hero-title .accent {
    color: #2F6FED;
}
.hero-sub {
    color: #64748B;
    font-size: 0.98rem;
    line-height: 1.55;
    margin-bottom: 1.8rem;
    max-width: 34ch;
}

/* Section divider label, e.g. "01 / RESUME" */
.section-label {
    font-weight: 700;
    font-size: 0.85rem;
    color: #0F172A;
    border-top: 1px solid #E5E7EB;
    padding-top: 1.1rem;
    margin-top: 0.4rem;
    margin-bottom: 0.6rem;
}

/* Report card */
.report-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 1.8rem 2rem 2rem 2rem;
    box-shadow: 0 1px 0 rgba(15, 23, 42, 0.02);
}
.report-topbar {
    height: 6px;
    background: linear-gradient(90deg, #2F6FED 0%, #2F6FED 100%);
    border-radius: 4px;
    margin: -1.8rem -2rem 1.6rem -2rem;
}
.report-header-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}
.report-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #0F172A;
    margin: 0.15rem 0 0.15rem 0;
}
.report-candidate {
    color: #64748B;
    font-size: 0.85rem;
}
.match-coefficient {
    text-align: right;
}
.match-coefficient .big-num {
    font-size: 2.4rem;
    font-weight: 800;
    color: #2F6FED;
    line-height: 1;
}
.match-coefficient .label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #94A3B8;
    letter-spacing: 0.05em;
}

.stat-row {
    display: flex;
    gap: 2.5rem;
    border-top: 1px solid #E5E7EB;
    border-bottom: 1px solid #E5E7EB;
    padding: 1.1rem 0;
    margin: 1.3rem 0 1.5rem 0;
}
.stat-item .stat-value {
    font-size: 1.3rem;
    font-weight: 800;
    color: #0F172A;
}
.stat-item .stat-value.blue { color: #2F6FED; }
.stat-item .stat-label {
    font-size: 0.72rem;
    color: #94A3B8;
    font-weight: 600;
    letter-spacing: 0.03em;
}

.target-position-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.3rem;
}
.target-position-summary {
    color: #64748B;
    font-size: 0.9rem;
    line-height: 1.5;
    margin-bottom: 1.4rem;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-bottom: 1.5rem;
}
.badge-pill {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1D4ED8;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 6px 12px;
    border-radius: 8px;
}

.gap-list {
    list-style: none;
    padding-left: 0;
    margin: 0 0 1.5rem 0;
}
.gap-list li {
    position: relative;
    padding-left: 18px;
    margin-bottom: 0.55rem;
    color: #334155;
    font-size: 0.9rem;
    line-height: 1.45;
}
.gap-list li::before {
    content: "";
    position: absolute;
    left: 0;
    top: 7px;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #DC2626;
}

.recommendation-box {
    background: #F8FAFC;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1.4rem;
}
.recommendation-box .rec-title {
    font-weight: 700;
    font-size: 0.85rem;
    color: #0F172A;
    margin-bottom: 0.4rem;
}
.recommendation-box .rec-body {
    color: #64748B;
    font-size: 0.88rem;
    line-height: 1.5;
}

/* Empty state */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 4.5rem 2rem;
    color: #94A3B8;
}
.empty-state .icon {
    font-size: 2rem;
    margin-bottom: 0.8rem;
}
.empty-state .title {
    font-weight: 700;
    color: #334155;
    font-size: 1.05rem;
    margin-bottom: 0.35rem;
}
.empty-state .sub {
    font-size: 0.88rem;
    max-width: 30ch;
}

/* Streamlit widget tweaks */
div[data-testid="stFileUploaderDropzone"] {
    border-radius: 10px;
}
div.stButton > button, div.stDownloadButton > button {
    background: #2F6FED;
    color: white;
    font-weight: 700;
    border-radius: 9px;
    border: none;
    padding: 0.6rem 1rem;
    letter-spacing: 0.01em;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    background: #2559C7;
    color: white;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="topbar">
        <div class="topbar-brand"><div class="icon-box">📄</div> REVIEW AGENT</div>
        <div class="topbar-status">SYSTEM STATUS: <span>READY</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_pdf_text(uploaded_file) -> str:
    """Extract text from an uploaded PDF file. Returns '' on failure."""
    try:
        reader = PdfReader(uploaded_file)
        text_parts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(text_parts).strip()
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


def parse_json_result(raw_text: str) -> dict:
    """Strip Markdown code fences (if any) and parse the agent's JSON output."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()
    return json.loads(cleaned)


def build_crew(resume_text: str, job_title: str, job_url: str, requirements: str) -> Crew:
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
            "Evaluate how well a candidate's resume matches a target role, "
            "being accurate and never inventing skills or experience that "
            "aren't actually present in the resume."
        ),
        backstory=(
            "You are a meticulous, experienced recruiter who has screened "
            "thousands of resumes. You give direct, honest, and constructive "
            "feedback. You never fabricate qualifications the candidate does "
            "not have, and you always ground your feedback strictly in the "
            "resume text and role requirements provided to you. You always "
            "respond with valid JSON only — no prose, no Markdown fences, no "
            "commentary before or after the JSON object."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "You will be given a RESUME and a set of TARGET ROLE REQUIREMENTS "
            f"below.\n\nTARGET ROLE TITLE: {job_title or 'Not specified'}\n\n"
            "RESUME:\n----------------\n"
            f"{resume_text}\n----------------\n\n"
            "TARGET ROLE REQUIREMENTS:\n----------------\n"
            f"{requirements}\n----------------\n\n"
            "Evaluate the resume against these requirements:\n"
            "1. Split the requirements into 'required qualifications' "
            "(must-haves: years of experience, specific languages/"
            "frameworks/tools, degrees, certifications) and 'additional "
            "qualifications' (nice-to-haves/preferred).\n"
            "2. Separately list any requirements that are inherently NOT "
            "verifiable from a resume (soft skills like communication, "
            "teamwork, stakeholder management, ability to work "
            "independently/remotely) as interview_evaluated_qualifications.\n"
            "3. For every required/additional qualification, check the "
            "resume for clear evidence. Mark 'matched': true only if the "
            "resume explicitly supports it or very clearly implies it; "
            "otherwise 'matched': false with a short, specific 'reason' "
            "(e.g. 'No mention of PHP experience'). Never mark something "
            "matched by assuming or inventing experience not stated in the "
            "resume.\n"
            "4. Compute match_percentage (0-100) as an honest overall fit "
            "score, and match_level as one of 'Low', 'Moderate', or 'High' "
            "(roughly: 0-40% = Low, 41-74% = Moderate, 75-100% = High).\n"
            "5. Write 3-5 genuine key_strengths, 3-5 specific gaps (plain "
            "phrases naming what's missing, no long justification), and 3-5 "
            "concrete, honest recommendations (never suggesting fabricating "
            "experience).\n"
            "6. Write one strategic_recommendation: the single most "
            "impactful piece of advice, as 1-2 sentences.\n\n"
            "Base every statement strictly on the text provided — do NOT "
            "assume or invent any skill, tool, or experience that isn't "
            "explicitly stated or clearly implied in the resume."
        ),
        expected_output=(
            "ONLY a single valid JSON object (no Markdown fences, no other "
            "text) with EXACTLY this shape:\n"
            "{\n"
            '  "match_level": "Low" | "Moderate" | "High",\n'
            '  "match_percentage": <integer 0-100>,\n'
            '  "summary": "<1-2 sentence overall summary>",\n'
            '  "required_qualifications": [\n'
            '    {"qualification": "<text>", "matched": <true|false>, "reason": "<empty string if matched>"}\n'
            "  ],\n"
            '  "additional_qualifications": [ <same shape as above, [] if none> ],\n'
            '  "interview_evaluated_qualifications": ["<text>", ...],\n'
            '  "key_strengths": ["<text>", ...],\n'
            '  "gaps": ["<text>", ...],\n'
            '  "recommendations": ["<text>", ...],\n'
            '  "strategic_recommendation": "<1-2 sentence single most impactful piece of advice>"\n'
            "}"
        ),
        agent=reviewer,
    )

    return Crew(agents=[reviewer], tasks=[task], process=Process.sequential, verbose=False)


def render_empty_state():
    st.markdown(
        """
        <div class="report-card">
            <div class="empty-state">
                <div class="icon">🗂️</div>
                <div class="title">No audit yet</div>
                <div class="sub">Fill in your resume and target role on the left,
                then generate your audit report.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_report(data: dict, candidate_label: str, job_title: str, job_url: str):
    required = data.get("required_qualifications", []) or []
    additional = data.get("additional_qualifications", []) or []
    interview_quals = data.get("interview_evaluated_qualifications", []) or []

    matched_required = [q for q in required if q.get("matched")]
    unmatched_required = [q for q in required if not q.get("matched")]
    matched_additional = [q for q in additional if q.get("matched")]
    unmatched_additional = [q for q in additional if not q.get("matched")]

    core_skills_met = f"{len(matched_required)}/{len(required)}" if required else "—"
    critical_gaps = len(unmatched_required)
    match_pct = data.get("match_percentage", 0)
    match_level = data.get("match_level", "—")

    badge_html = "".join(
        f'<span class="badge-pill">✓ {q["qualification"]}</span>'
        for q in (matched_required + matched_additional)
    )
    if not badge_html:
        badge_html = '<span class="badge-pill">No competencies matched yet</span>'

    gaps_for_card = data.get("gaps", []) or [q["qualification"] for q in (unmatched_required + unmatched_additional)]
    gap_items_html = "".join(f"<li>{g}</li>" for g in gaps_for_card[:5])

    job_title_display = job_title.strip() if job_title.strip() else "Target Role"
    job_link_html = (
        f'<a href="{job_url}" target="_blank" style="color:#2F6FED;text-decoration:none;">{job_title_display}</a>'
        if job_url.strip()
        else job_title_display
    )

    st.markdown(
        f"""
        <div class="report-card">
            <div class="report-topbar"></div>
            <div class="eyebrow-plain">ASSESSMENT REPORT · 01</div>
            <div class="report-header-row">
                <div>
                    <div class="report-title">Resume Audit</div>
                    <div class="report-candidate">CANDIDATE: {candidate_label}</div>
                </div>
                <div class="match-coefficient">
                    <div class="big-num">{match_pct}%</div>
                    <div class="label">MATCH COEFFICIENT</div>
                </div>
            </div>

            <div class="stat-row">
                <div class="stat-item">
                    <div class="stat-value">{core_skills_met}</div>
                    <div class="stat-label">CORE SKILLS MET</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value blue">{critical_gaps:02d}</div>
                    <div class="stat-label">CRITICAL GAPS</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{match_level}</div>
                    <div class="stat-label">MATCH LEVEL</div>
                </div>
            </div>

            <div class="eyebrow-plain">TARGET POSITION</div>
            <div class="target-position-title">{job_link_html}</div>
            <div class="target-position-summary">{data.get("summary", "")}</div>

            <div class="eyebrow-plain">MATCHED COMPETENCIES</div>
            <div class="badge-row">{badge_html}</div>

            <div class="eyebrow-danger">DETECTED GAPS</div>
            <ul class="gap-list">{gap_items_html}</ul>

            <div class="recommendation-box">
                <div class="rec-title">STRATEGIC RECOMMENDATION</div>
                <div class="rec-body">{data.get("strategic_recommendation", "")}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_export, col_detail = st.columns([1, 1])
    with col_export:
        report_markdown = build_full_report_markdown(data, candidate_label, job_title)
        st.download_button(
            "⬇️  EXPORT REPORT",
            data=report_markdown,
            file_name="resume_audit_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_detail:
        with st.expander("VIEW DETAILED GUIDANCE →"):
            st.markdown("**Required qualifications**")
            for q in required:
                mark = "✓" if q.get("matched") else "?"
                extra = f" _({q.get('reason')})_" if not q.get("matched") and q.get("reason") else ""
                st.markdown(f"- {mark} {q['qualification']}{extra}")

            if additional:
                st.markdown("**Additional (preferred) qualifications**")
                for q in additional:
                    mark = "✓" if q.get("matched") else "?"
                    extra = f" _({q.get('reason')})_" if not q.get("matched") and q.get("reason") else ""
                    st.markdown(f"- {mark} {q['qualification']}{extra}")

            if interview_quals:
                st.markdown("**Typically evaluated in the application/interview**")
                for q in interview_quals:
                    st.markdown(f"- • {q}")

            st.markdown("**Key strengths**")
            for s in data.get("key_strengths", []):
                st.markdown(f"- {s}")

            st.markdown("**All gaps**")
            for g in data.get("gaps", []):
                st.markdown(f"- {g}")

            st.markdown("**Recommendations**")
            for i, r in enumerate(data.get("recommendations", []), 1):
                st.markdown(f"{i}. {r}")


def build_full_report_markdown(data: dict, candidate_label: str, job_title: str) -> str:
    required = data.get("required_qualifications", []) or []
    additional = data.get("additional_qualifications", []) or []
    lines = [
        f"# Resume Audit — {candidate_label}",
        f"**Target role:** {job_title or 'Not specified'}",
        f"**Match level:** {data.get('match_level', '—')}  |  **Match coefficient:** {data.get('match_percentage', 0)}%",
        "",
        f"> {data.get('summary', '')}",
        "",
        "## Required Qualifications",
    ]
    for q in required:
        mark = "✓" if q.get("matched") else "?"
        extra = f" ({q.get('reason')})" if not q.get("matched") and q.get("reason") else ""
        lines.append(f"- {mark} {q['qualification']}{extra}")

    if additional:
        lines.append("\n## Additional (Preferred) Qualifications")
        for q in additional:
            mark = "✓" if q.get("matched") else "?"
            extra = f" ({q.get('reason')})" if not q.get("matched") and q.get("reason") else ""
            lines.append(f"- {mark} {q['qualification']}{extra}")

    if data.get("interview_evaluated_qualifications"):
        lines.append("\n## Typically Evaluated in the Application/Interview")
        for q in data["interview_evaluated_qualifications"]:
            lines.append(f"- {q}")

    lines.append("\n## Key Strengths")
    for s in data.get("key_strengths", []):
        lines.append(f"- {s}")

    lines.append("\n## Gaps")
    for g in data.get("gaps", []):
        lines.append(f"- {g}")

    lines.append("\n## Recommendations")
    for i, r in enumerate(data.get("recommendations", []), 1):
        lines.append(f"{i}. {r}")

    lines.append("\n## Strategic Recommendation")
    lines.append(data.get("strategic_recommendation", ""))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layout — two columns: intake (left) / report (right)
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown('<div class="eyebrow-pill">PHASE 01 · CANDIDATE INTAKE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Review<br><span class="accent">Agent.</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">Compare your experience against a target role '
        "and receive a clear, structured assessment of your market fit.</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">01 / RESUME</div>', unsafe_allow_html=True)
    resume_input_mode = st.radio(
        "Resume input",
        ["Upload file", "Paste text"],
        horizontal=True,
        label_visibility="collapsed",
    )

    resume_text = ""
    candidate_label = "Pasted Resume"
    if resume_input_mode == "Upload file":
        uploaded_pdf = st.file_uploader(
            "Choose a resume file", type=["pdf"], help="PDF up to 10MB", label_visibility="visible"
        )
        if uploaded_pdf is not None:
            candidate_label = uploaded_pdf.name.upper()
            with st.spinner("Extracting text from PDF..."):
                resume_text = extract_pdf_text(uploaded_pdf)
            if resume_text:
                with st.expander("Preview extracted resume text"):
                    st.text(resume_text[:3000] + ("..." if len(resume_text) > 3000 else ""))
    else:
        resume_text = st.text_area("Paste resume text", height=180, label_visibility="collapsed")

    st.markdown('<div class="section-label">02 / TARGET ROLE</div>', unsafe_allow_html=True)
    col_title, col_url = st.columns(2)
    with col_title:
        job_title = st.text_input("Job title", placeholder="AI Engineer")
    with col_url:
        job_url = st.text_input("Posting URL", placeholder="LinkedIn / Indeed")

    requirements = st.text_area(
        "Requirements description",
        placeholder=(
            "Building web-based applications using PHP for backend "
            "development and React for frontend interfaces. Designing and "
            "managing robust SQL databases. Integrating AI platforms and "
            "developing scalable AWS cloud solutions."
        ),
        height=140,
    )

    run_button = st.button("✨  GENERATE AUDIT REPORT", use_container_width=True)

with right:
    if not run_button:
        render_empty_state()
    else:
        if not resume_text or not resume_text.strip():
            st.warning("⚠️ Please provide a resume (upload a PDF or paste text) before running.")
            render_empty_state()
        elif not requirements or not requirements.strip():
            st.warning("⚠️ Please describe the target role's requirements before running.")
            render_empty_state()
        else:
            try:
                with st.spinner("Auditing resume against target role... this can take up to a minute."):
                    crew = build_crew(resume_text, job_title, job_url, requirements)
                    result = crew.kickoff()
                    data = parse_json_result(str(result))

                render_report(data, candidate_label, job_title, job_url)

            except json.JSONDecodeError:
                st.error(
                    "⚠️ The model's response wasn't valid JSON, so the report "
                    "couldn't be rendered. Try running the audit again."
                )
                with st.expander("Show raw model output"):
                    st.text(str(result))
                render_empty_state()

            except Exception as e:
                error_message = str(e).lower()
                if "rate limit" in error_message or "429" in error_message:
                    st.error(
                        "⚠️ Groq's API rate limit was hit. Please wait a moment "
                        "and try again."
                    )
                elif "api key" in error_message or "401" in error_message or "unauthorized" in error_message:
                    st.error(
                        "⚠️ Groq rejected the API key. Double-check "
                        "`GROQ_API_KEY` in your Streamlit secrets."
                    )
                elif "timeout" in error_message:
                    st.error("⚠️ The request to Groq timed out. Please try again.")
                else:
                    st.error(f"⚠️ Something went wrong while generating the audit.\n\nDetails: {e}")
                render_empty_state()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "Built with CrewAI, Streamlit, and Groq (openai/gpt-oss-120b). "
    "This tool provides guidance only and does not guarantee interview or job outcomes."
)
