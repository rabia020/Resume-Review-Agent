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
└── .streamlit/
    └── secrets.toml.example        # Template for your Groq API key
```

That's it — only 3 real files. Minimal, so it's easy to upload to GitHub
and deploy.

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
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then open .streamlit/secrets.toml and paste your real key in

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
   - **Do not commit a real `secrets.toml`** — only the `.example` file
     should go to GitHub. Add `.streamlit/secrets.toml` to a `.gitignore`
     if you created one locally.

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

1. **You provide input**: a resume (pasted text or PDF upload) and a job
   description (pasted text).
2. If you uploaded a PDF, the app extracts its text using `pypdf`.
3. A single CrewAI **Agent** (acting as a "Senior Technical Recruiter") is
   given both texts and asked to:
   - Estimate a match score
   - List genuine strengths
   - List real gaps/missing qualifications
   - Suggest concrete, honest improvements
4. The agent is explicitly instructed **not to invent** any skill or
   experience not actually present in the resume.
5. The result is displayed as a formatted report in the browser.

---
## 6. Live Demo
[Try Live App](https://resume-review-agent-app.streamlit.app/)

## 7. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| "No Groq API key found" | Add `GROQ_API_KEY` to `.streamlit/secrets.toml` (local) or the Secrets panel (cloud). |
| "Groq rejected the API key" | Double check you copied the full key with no extra spaces, and that it's still active in the Groq console. |
| "Groq's API rate limit was hit" | Wait a bit and try again, or check your usage limits at console.groq.com. |
| "I couldn't read that PDF" | The PDF may be a scanned image with no selectable text. Try "Paste text" instead. |
| App works locally but not on Streamlit Cloud | Make sure the Python version is set to 3.11 in Advanced settings, and that secrets were saved on the Cloud dashboard (not just locally). |

---

## 7. Notes for customization

- To change the LLM model, edit the `model=` line inside `build_crew()` in
  `app.py` — just make sure it's a valid, active Groq production model.
- To change what the report covers, edit the `expected_output` text inside
  the `Task(...)` definition in `app.py`.
