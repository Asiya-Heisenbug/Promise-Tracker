"""
app.py - Streamlit interface for JobLens.   Run with:  streamlit run app.py

Flow: preferences -> paste job -> Analyze -> summary / plain-language / skills / fit score -> chat.
Everything lives in st.session_state (no database).
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from analyzer import analyze_reviews, fit_score, plain_language
from chatbot import answer
from nlp import engine_status, entities, extract_job
from scraper import fetch_page_text, get_company_info

SAMPLE = Path(__file__).parent / "sample_data"

st.set_page_config(page_title="JobLens", page_icon="🔍", layout="wide")
st.markdown("<style>.block-container{max-width:1100px;padding-top:2rem;}</style>", unsafe_allow_html=True)

for key in ("jd_text", "job_url", "fetch_msg", "company_url", "company_text", "reviews_text", "review_source"):
    st.session_state.setdefault(key, "")
st.session_state.setdefault("chat", [])


def load_sample():
    read = lambda f: (SAMPLE / f).read_text(encoding="utf-8")  # noqa: E731
    st.session_state.jd_text = read("sample_job.txt")
    st.session_state.company_text = read("sample_company.txt")
    st.session_state.reviews_text = read("sample_reviews.txt")
    st.session_state.review_source = "Sample data (fictional reviews written for this demo)"


def fetch_job():
    text, err = fetch_page_text(st.session_state.job_url)
    st.session_state.fetch_msg = err or "Page text loaded. Check it in the 'Paste text' tab."
    if text:
        st.session_state.jd_text = text


def show(value, default="Not mentioned"):
    return value if value else default


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("NLP engines")
    for name, on in engine_status().items():
        st.write(("✅ " if on else "⚪ ") + name)
    st.caption("Greyed-out engines are optional; JobLens falls back to simpler methods.")

# ---------------------------------------------------------------- header
st.title("🔍 JobLens")
st.caption("AI Job Intelligence & Fit Analyzer: paste a job, see how it fits *you*.")

# ---------------------------------------------------------------- 1. preferences
st.header("1 · My Preferences")
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    role = c1.text_input("Preferred role", "AI Developer")
    location = c2.text_input("Preferred location", "Chennai")
    min_salary = c3.number_input("Minimum salary (₹ / month)", 0, 1_000_000, 40_000, 5_000)
    shift = c1.selectbox("Preferred shift", ["Any", "Morning", "Evening", "Night"], index=1)
    experience = c2.text_input("Experience", "Fresher")
    skills = c3.text_input("Skills (comma-separated)", "Python, ML, NLP, FastAPI")
prefs = {"role": role, "location": location, "min_salary": int(min_salary),
         "shift": shift, "experience": experience, "skills": skills}

# ---------------------------------------------------------------- 2. job input
st.header("2 · Paste Job Description")
st.button("Load sample job (demo)", on_click=load_sample)
tab_paste, tab_url = st.tabs(["Paste text", "Job URL (optional)"])
with tab_paste:
    st.text_area("Job description", key="jd_text", height=260, label_visibility="collapsed",
                 placeholder="Paste the full job description here...")
with tab_url:
    st.text_input("Job page URL", key="job_url", placeholder="https://...")
    st.button("Fetch page text", on_click=fetch_job)
    if st.session_state.fetch_msg:
        st.info(st.session_state.fetch_msg)
    st.caption("Many job sites block automated access or load text with JavaScript. If fetching fails, just paste the text.")
with st.expander("Optional: company info & employee reviews (paste public text)"):
    st.text_input("Company website (optional)", key="company_url", placeholder="https://...")
    st.text_area("Company information", key="company_text", height=90,
                 placeholder="Paste a short public description of the company. Or leave empty and JobLens will try Wikipedia.")
    st.text_area("Employee reviews: separate each review with a blank line", key="reviews_text", height=150)
    st.text_input("Where are these reviews from?", key="review_source", placeholder="e.g. AmbitionBox, copied on 7 Oct 2026")
    st.caption("JobLens does not scrape review sites (their terms usually forbid it). Paste a few reviews you can see yourself.")

# ---------------------------------------------------------------- 3. analyze
st.header("3 · Analyze Job")
if st.button("Analyze job", type="primary"):
    if len(st.session_state.jd_text.strip()) < 40:
        st.error("Please paste a job description first.")
    else:
        with st.spinner("Reading the job description..."):
            job = extract_job(st.session_state.jd_text)
            st.session_state.result = {
                "prefs": prefs, "job": job,
                "fit": fit_score(prefs, job), "plain": plain_language(job),
                "company": get_company_info(job["company"], st.session_state.company_url,
                                            st.session_state.company_text),
                "reviews": analyze_reviews(st.session_state.reviews_text, st.session_state.review_source),
            }
            st.session_state.chat = []

res = st.session_state.get("result")
if res:
    job, fit, plain = res["job"], res["fit"], res["plain"]

    # ------------------------------------------------------------ 4. summary
    st.header("4 · Job Summary")
    with st.container(border=True):
        cols = st.columns(4)
        cols[0].markdown(f"**Title**  \n{show(job['title'])}")
        cols[1].markdown(f"**Company**  \n{show(job['company'])}")
        cols[2].markdown(f"**Location**  \n{show(job['location'])}")
        cols[3].markdown(f"**Salary**  \n{show(job['salary'] and job['salary']['text'])}")
        cols = st.columns(4)
        cols[0].markdown(f"**Experience**  \n{show(job['experience'] and job['experience']['text'])}")
        cols[1].markdown(f"**Work mode**  \n{show(job['work_mode'])}")
        cols[2].markdown(f"**Shift**  \n{show(job['shift'])}")
        if job["salary"]:
            cols[3].markdown(f"**Per month (approx.)**  \n₹{job['salary']['min_monthly']:,} – ₹{job['salary']['max_monthly']:,}")
    with st.expander("Under the hood: named entities found by spaCy"):
        ents = entities(job["raw_text"])
        if ents:
            st.dataframe(pd.DataFrame(ents, columns=["Text", "Type"]), hide_index=True)
        else:
            st.write("spaCy isn't installed (or found nothing). Fields above come from regex + keyword rules.")

    # ------------------------------------------------------------ 5. plain language
    st.header("5 · What You'll Actually Do")
    st.info(plain["summary"])
    if plain["by_llm"]:
        st.caption("Rewritten by the LLM from the extracted responsibilities.")
    for b in plain["bullets"]:
        st.markdown(f"- {b}")
    if plain["translations"]:
        with st.expander("Corporate wording → plain English"):
            st.dataframe(pd.DataFrame(plain["translations"], columns=["Job description says", "In plain English"]),
                         hide_index=True)

    # ------------------------------------------------------------ 6. skills
    st.header("6 · Required Skills")
    sk = fit["skills"]
    icon = lambda s: "✅" if s in sk["mine"] else "🟡" if s in sk["related"] else "❌"  # noqa: E731
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**Required**")
        st.markdown("  \n".join(f"{icon(s)} `{s}`" for s in job["required_skills"]) or "None detected")
    with c2, st.container(border=True):
        st.markdown("**Nice to have**")
        st.markdown("  \n".join(f"{icon(s)} `{s}`" for s in job["preferred_skills"]) or "None detected")
    st.caption("✅ you list it · 🟡 related to a skill you have (semantic match) · ❌ not in your skills")

    # ------------------------------------------------------------ 7. fit score
    st.header("7 · Personal Fit Score")
    with st.container(border=True):
        if fit["percent"] is None:
            st.warning("There isn't enough information in this job description to estimate a fit score.")
        else:
            c1, c2 = st.columns([1, 3])
            c1.metric("Estimated Job Fit", f"{fit['percent']}%")
            c1.write(fit["label"])
            c2.progress(fit["percent"] / 100)
            c2.markdown(fit["explanation"])
        rows = [{"Criterion": c["name"], "Weight": f"{c['weight']}%",
                 "Result": "➖" if c["score"] is None else "✅" if c["score"] >= 0.8 else "🟡" if c["score"] >= 0.4 else "❌",
                 "Details": c["note"]} for c in fit["checks"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(f"⚠️ This is an *estimated* fit score: a simple weighted comparison using {fit['criteria_used']} criteria "
                   "the job description covers. It is not scientifically validated.")

    # ------------------------------------------------------------ 8. company
    st.header("8 · Company Information")
    co = res["company"]
    with st.container(border=True):
        if co:
            st.markdown(f"**{show(job['company'])}**")
            st.write(co["description"])
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Industry**  \n{show(co['industry'])}")
            c2.markdown(f"**Location**  \n{show(co['location'] or job['location'])}")
            src = co["source"]
            c3.markdown(f"**Source**  \n{f'[{src}]({src})' if src.startswith('http') else src}")
            st.caption("Basic public information only. Please check it's the right company.")
        else:
            if job["company"]:
                st.info(f"No matching public company summary was found for **{job['company']}**. "
                        "JobLens searched Wikipedia. Add the official website or paste a company description "
                        "under *Optional: company info & employee reviews*, then analyze again.")
            else:
                st.info("No company name was detected in this job description, so JobLens could not search for "
                        "the company. Add a line such as `Company: Acme` or enter its website under *Optional: "
                        "company info & employee reviews*, then analyze again.")
            st.caption("Employee reviews are not searched or scraped; paste reviews you can access and their source.")

    # ------------------------------------------------------------ 9. reviews
    st.header("9 · Review Analysis")
    rv = res["reviews"]
    with st.container(border=True):
        if rv:
            st.markdown(f"Among the **{rv['n']} available review(s)**, these themes appeared most often. "
                        f"Source: {rv['source'] or 'not provided'}.")
            c1, c2 = st.columns(2)
            for col, title, side in ((c1, "👍 Positive themes", "positive"), (c2, "⚠️ Concerns raised", "negative")):
                with col:
                    st.markdown(f"**{title}**")
                    for t in rv[side]:
                        st.markdown(f"- **{t['theme']}**: {t['count']} of {rv['n']} reviews  \n  _“{t['example']}”_")
                    if not rv[side]:
                        st.write("Nothing stood out.")
            st.caption("Simple keyword + sentiment-word analysis on a small sample. It describes these reviews only, "
                       "not the company as a whole.")
        else:
            st.info("No reviews provided. Paste a few public reviews (and where they're from) under *Optional: "
                    "company info & employee reviews* above.")

    # ------------------------------------------------------------ 10. chatbot
    st.header("10 · Ask JobLens")
    quick = ["Will this job suit me?", "What skills am I missing?", "What will I actually do?", "Should I apply?"]
    question = None
    for col, q in zip(st.columns(len(quick)), quick):
        if col.button(q, key=f"quick_{q}"):
            question = q
    question = st.chat_input("Ask about this job or company...") or question
    if question:
        st.session_state.chat += [("user", question), ("assistant", answer(question, res))]
    for who, msg in st.session_state.chat:
        with st.chat_message(who):
            st.markdown(msg)
else:
    st.info("Fill in your preferences, paste (or load) a job description, then click **Analyze job**.")
