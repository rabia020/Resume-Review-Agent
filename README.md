# Resume Review Agent

A beginner-friendly, single-agent AI app that compares a candidate's resume
against a target job description and gives structured, honest feedback —
without inventing skills the candidate doesn't have.

Built with:
- **CrewAI** — runs the single review agent
- **Groq** (`openai/gpt-oss-120b`) — the LLM powering the agent
- **Streamlit** — the web interface

---

## 1. File structure

```
resume-review-agent/
├── app.py                          # The whole app (UI + agent logic)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Keeps your real API key out of GitHub
└── .streamlit/
    ├── secrets.toml                # Your real Groq API key (local only — never commit)
    └── secrets.toml.example        # Template for your Groq API key
```

Minimal by design, so it's easy to upload to GitHub and deploy.

---

## 2. Get a free Groq API key

1. Go to https://console.groq.com and sign up (it's free).
2. Create an API key from the dashboard.
3. Keep it handy for step 4 below — never paste it directly into `app.py`.

---

## 3. Run it locally (optional, but good for testing)

```bash
# 1. Clone or download this folder, then cd into it
cd resume-review-agent

# 2. Create a virtual environment (Python 3.11 recommended)
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq key locally
# .streamlit/secrets.toml already exists as a template —
# just open it and replace the placeholder with your real key.

# 5. Run the app
streamlit run app.py
```

Your browser should open to `http://localhost:8501`.

---

## 4. Deploy to Streamlit Community Cloud (free)

1. Push this folder to a **new GitHub repository**.
   - Make sure `app.py`, `requirements.txt`, and the `.streamlit` folder
     structure are at the **root** of the repo (or note the sub-path when
     deploying).
   - **Do not commit a real `secrets.toml`** — the included `.gitignore`
     already excludes `.streamlit/secrets.toml` so this happens
     automatically. Only `.streamlit/secrets.toml.example` should be visible
     on GitHub.

2. Go to https://share.streamlit.io and sign in with GitHub.

3. Click **"New app"**, then:
   - Select your repository and branch.
   - Set **Main file path** to `app.py`.
   - Under **Advanced settings**, set the Python version to **3.11**.

4. Before clicking Deploy, go to **Settings → Secrets** (or do this right
   after the first deploy) and paste:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
   Save — the app will restart automatically with the key available.

5. Click **Deploy**. After a minute or two, you'll get a public URL you can
   share.

---

## 5. How it works (plain-English overview)

1. **You provide input**: a resume (pasted text or PDF upload), a job
   description (pasted text), and optionally the job title and posting URL.
2. If you uploaded a PDF, the app extracts its text using `pypdf`.
3. A single CrewAI **Agent** (acting as a "Senior Technical Recruiter") is
   given all of this and asked to:
   - Split the job description into **required** and **additional
     (preferred)** qualifications, plus a separate list of soft-skill
     qualifications that are normally judged in an interview rather than
     from a resume (communication, teamwork, remote-work fit, etc.)
   - Check each required/additional qualification against the resume and
     mark it **✓** (matched) or **?** (unmatched, with a specific reason —
     e.g. "No mention of PHP experience")
   - Produce a LinkedIn "Job Match"-style summary at the top of the report
     (`Job match is Low/Medium/High - [Job Title](url)`), followed by the
     full checklist
   - Then add an overall match score, genuine strengths, real gaps, and
     concrete, honest improvement suggestions
4. The agent is explicitly instructed **not to invent** any skill or
   experience not actually present in the resume, and to only mark a
   qualification as matched when the resume clearly supports it.
5. The result is displayed as a formatted report in the browser.

---

## 6. Live Demo

[Try Resume Review App Live](https://resume-review-agent-app.streamlit.app/)


---

## 7. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| "No Groq API key found" | Add `GROQ_API_KEY` to `.streamlit/secrets.toml` (local) or the Secrets panel (cloud). |
| "Groq rejected the API key" | Double check you copied the full key with no extra spaces, and that it's still active in the Groq console. |
| "Groq's API rate limit was hit" | Wait a bit and try again, or check your usage limits at console.groq.com. |
| "I couldn't read that PDF" | The PDF may be a scanned image with no selectable text. Try "Paste text" instead. |
| "Unable to initialize LLM..." / provider errors | Make sure `app.py`'s `LLM(...)` call uses `provider="openai"` with `base_url="https://api.groq.com/openai/v1"` — CrewAI's newer versions no longer auto-detect Groq via a `groq/` prefix. |
| App works locally but not on Streamlit Cloud | Make sure the Python version is set to 3.11 in Advanced settings, and that secrets were saved on the Cloud dashboard (not just locally). |

---

## 8. Notes for customization

- To change the LLM model, edit the `model=` and `base_url=` lines inside
  `build_crew()` in `app.py` — just make sure it's a valid, active Groq
  production model, and that the base URL matches Groq's OpenAI-compatible
  endpoint.
- To change what the qualification checklist or report covers, edit the
  `description` and `expected_output` text inside the `Task(...)`
  definition in `app.py`.

---
