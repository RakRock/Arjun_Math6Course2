## 6th Grade Springboard Course 2 Quiz Generator

Streamlit app that ingests Springboard Course 2 PDFs, extracts keyword frequencies, generates weighted quizzes with xAI Grok, grades answers, and tracks progress in SQLite.

### Quick Start
```bash
cd /Users/rmudisoo/Documents/AWS/Arjun/MathProd
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install streamlit openai pdfplumber plotly pandas
export XAI_API_KEY="your_grok_key"   # add to ~/.zshrc for convenience
streamlit run app.py
```

### Using the App
- Drop unit PDFs into `pdf_units/` then click **Refresh PDFs from folder** to rebuild `units_data.json`.
- Enter **Student Name**, choose units, and pick a game mode (Rookie Quest / Adventurer Quest / Boss Battle).
- Click **Generate Quiz** (20 weighted questions). Submit once—answers lock after grading.
- View progress via sidebar: heatmap, per-unit averages, and quiz history (mode, score, feedback).

### Data & Storage
- Keywords: `units_data.json`
- Quiz records: `progress.db` (SQLite). The app auto-migrates to add new columns like `difficulty`.
- Temp PDFs cleaned after processing.

### Troubleshooting
- Missing plotly/pandas? Re-run the pip install line above.
- Grok issues: confirm `XAI_API_KEY` and network access; quiz generation errors show raw responses when debug is checked.
- Fewer/more than 20 questions: the app trims or warns; responses are logged in session state for debugging.

### Notes
- Weighted question mix is derived from keyword counts per selected unit (higher frequency ⇒ more emphasis).
- Feedback lists “needs improvement” items first, then strengths.
# Arjun_Math6Course2
LLM based exercise generator for Course2Grade6 Springboard
