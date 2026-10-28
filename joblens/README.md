# 🔍 JobLens: AI Job Intelligence & Fit Analyzer

A small college NLP prototype. **Paste a job → JobLens understands it → compares it with *your* preferences → explains it in plain English → lets you ask questions.**

No database, no Docker, no auth. Streamlit + a few short Python files.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm                # optional: entity extraction
streamlit run app.py
```

The project disables Streamlit's file watcher to avoid a PyTorch `torch.classes` watcher traceback on Windows. Restart the app manually after changing code.

Click **Load sample job (demo)** → **Analyze job**. The sample works fully offline.

*Notes:* the first run downloads the `all-MiniLM-L6-v2` sentence-transformers model (~90 MB). If spaCy or sentence-transformers are missing, JobLens still runs with simpler fallbacks (check the sidebar to see which engines are active).

When a company name is detected and no company description is pasted, JobLens searches Wikipedia for a matching public company summary and shows the source link. Add the official company website for its meta description if it has no Wikipedia article. JobLens does not search or scrape employee-review sites; paste reviews and their source yourself.

**Optional LLM (Gemini):** to enable rewrites and answers to questions the rule-based chatbot doesn't recognise, create a local `.env` file in the `joblens` folder:

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace `paste-your-gemini-api-key-here` with your Gemini API key, save the file, and restart Streamlit. The sidebar should show Gemini as active. Keep `.env` private; it is excluded from Git. Do not paste your key into source code, README files, screenshots, or chat.

For one PowerShell session instead, set `$env:GEMINI_API_KEY="your-key"` before running `streamlit run app.py`. In hosted deployments, add `GEMINI_API_KEY` as a platform secret/environment variable. Common questions about fit, skills, salary, location, shift, experience, responsibilities, company information, and reviews are answered from the analysis data without a key.

## Project structure

| File | What it does |
|---|---|
| `app.py` | Streamlit UI: the 10 sections, session state only |
| `nlp.py` | Extraction: regex (salary, experience, shift, location, work mode), skill vocabulary with aliases, section splitting, spaCy NER, sentence-transformers similarity, optional Gemini call |
| `analyzer.py` | Fit score (6 weighted criteria), responsibility-grounded "What will I actually do?" summary, review theme analysis |
| `chatbot.py` | Answers common job questions directly from extracted analysis data; says *"I couldn't find enough information to determine that."* when a fact is missing |
| `scraper.py` | Polite page fetch (robots.txt, JSON-LD), tiny company lookup (website meta description / Wikipedia / pasted text) |
| `sample_data/` | Demo job, company text and **fictional** reviews |

## NLP techniques (for your report)

| Task | Technique |
|---|---|
| Salary, experience, shift, location, work mode | Regular expressions (handles `₹40,000 per month`, `4-6 LPA`, `0-1 years`, `9 to 6`) |
| Required vs preferred skills | Section/heading detection + skill vocabulary with aliases (`ML` → `machine learning`) |
| Company / city fallback | spaCy named-entity recognition |
| Role match, related skills | Sentence embeddings (MiniLM) + cosine similarity; word-overlap fallback |
| Plain-language rewrite | Jargon dictionary + activity patterns; optional LLM rewrite grounded in extracted text |
| Review themes | Theme keywords + small sentiment lexicon with negation handling, applied per clause |
| Chatbot | Intent keywords → answers built from analysis data; LLM only as a grounded fallback |

No custom model is trained.

## How the fit score works

Skills 30 · Role 20 · Experience 15 · Salary 15 · Location 10 · Shift 10. Criteria the job doesn't mention are skipped and the weights re-scaled (missing data is never counted as a "no"). With fewer than 3 usable criteria no percentage is shown. **It is an estimate from a simple weighted comparison, not a validated prediction**, and the UI says so.

## Responsible use and limits

- JobLens respects `robots.txt` and gives up politely if a site blocks it; it then asks you to paste the text. There are no workarounds.
- It does **not** scrape review sites (their terms usually forbid it). You paste a few public reviews and say where they came from; results read "Among the available reviews, these themes appeared…", never "this company is bad".
- Regex rules are tuned for typical Indian job posts and may miss unusual formats. Skill vocabulary is a short list in `nlp.py`; add your own.
- Wikipedia lookup can match the wrong company, so always check the source link.

## Possible next steps

Larger skill vocabulary · compare several jobs side by side · export a report · a proper evaluation set of labelled job posts.
