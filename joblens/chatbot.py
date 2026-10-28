"""
chatbot.py - answers questions about the analysed job using ONLY collected data.

Common questions are answered by simple rules straight from the analysis results
(so nothing can be invented). Only unrecognised questions go to the optional LLM,
and the LLM is told to answer strictly from the provided context.
"""
import json
import re

from nlp import ask_llm

NO_INFO = "I couldn't find enough information to determine that."


def _check(ctx, name):
    return next(c for c in ctx["fit"]["checks"] if c["name"] == name)


def _field(ctx, name):
    job = ctx["job"]
    values = {
        "Salary": (job.get("salary") or {}).get("text"),
        "Location": job.get("location") or job.get("work_mode"),
        "Shift": job.get("shift"),
        "Experience": (job.get("experience") or {}).get("text"),
    }
    c = _check(ctx, name)
    value = values[name]
    if value:
        if c["score"] is not None:
            return f"{value}\n\n{c['note']}"
        return f"The job description lists: {value}."
    return c["note"] if c["score"] is not None else f"{c['note']} {NO_INFO}"


def _fit(ctx):
    fit = ctx["fit"]
    if fit["percent"] is None:
        out = ["I can't provide a reliable overall fit percentage because fewer than three criteria had enough information."]
        if fit["explanation"]:
            out.append(fit["explanation"])
        return "\n\n".join(out)
    return f"Your estimated fit is **{fit['percent']}%** ({fit['label']}).\n\n{fit['explanation']}"


def _skills(ctx):
    s, job = ctx["fit"]["skills"], ctx["job"]
    if not (job["required_skills"] or job["preferred_skills"]):
        return NO_INFO
    out = []
    if s["matched"]:
        out.append("You already list: " + ", ".join(s["matched"]) + ".")
    if s["related"]:
        out.append("Related skills you have: " + ", ".join(f"{k} (you have {v})" for k, v in s["related"].items()) + ".")
    out.append("Missing required skills: " + (", ".join(s["missing_required"]) or "none") + ".")
    out.append("Missing nice-to-have skills: " + (", ".join(s["missing_preferred"]) or "none") + ".")
    return "\n\n".join(out)


def _do(ctx):
    plain = ctx["plain"]
    if not ctx["job"]["responsibilities"] and not plain["bullets"]:
        return NO_INFO
    bullets = "\n".join(f"- {b}" for b in plain["bullets"])
    return plain["summary"] + ("\n\n" + bullets if bullets else "")


def _company(ctx):
    co, rv, out = ctx.get("company"), ctx.get("reviews"), []
    if co and co.get("description"):
        out.append(f"**About the company:** {co['description']} (source: {co.get('source') or 'user-provided'})")
    if rv:
        pos = ", ".join(f"{t['theme']} ({t['count']})" for t in rv["positive"][:4]) or "none stood out"
        neg = ", ".join(f"{t['theme']} ({t['count']})" for t in rv["negative"][:4]) or "none stood out"
        out.append(f"Among the {rv['n']} available review(s) ({rv['source'] or 'source not given'}), "
                   f"these themes appeared most often.\n\n- Positive: {pos}\n- Concerns: {neg}\n\n"
                   "This is a small sample, so please don't treat it as a verdict on the company.")
    return "\n\n".join(out) or NO_INFO


def _apply(ctx):
    fit = ctx["fit"]
    if fit["percent"] is None:
        return "I can't give a reliable apply recommendation yet.\n\n" + _fit(ctx)
    p, s = fit["percent"], fit["skills"]
    verdict = ("It looks worth applying: the match is strong." if p >= 75 else
               "It could be worth applying, but check the gaps first." if p >= 50 else
               "It looks like a weak match for your preferences right now.")
    out = [f"{verdict} (estimated fit: {p}%)"]
    if s["missing_required"]:
        out.append("Skills to brush up on first: " + ", ".join(s["missing_required"]) + ".")
    weak = [c["name"].lower() for c in fit["checks"] if c["score"] is not None and c["score"] < 0.5]
    if weak:
        out.append("Weak areas: " + ", ".join(weak) + ".")
    if ctx.get("reviews"):
        out.append("I also found review themes, so ask me about the company before deciding.")
    out.append("This is only an estimate. The final decision is yours.")
    return "\n\n".join(out)


def _help(ctx):
    title = ctx["job"].get("title")
    job_label = f" for **{title}**" if title else ""
    return (f"I can answer questions about the job analysis{job_label}, your preferences, and any company "
            "or review information you provided. Try asking about the salary, location, shift, "
            "experience, skills you are missing, day-to-day responsibilities, fit, or whether "
            "the job may be worth applying for.")


def _normalize(question):
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", question.casefold())).strip()


def _matches(question, keyword):
    if keyword.endswith("*"):
        prefix = keyword[:-1]
        return any(token.startswith(prefix) for token in question.split())
    return f" {keyword} " in f" {question} "


# (keywords, handler): the first match wins, so more specific topics come first
INTENTS = [
    (["should i apply", "worth it", "worth applying", "apply*"], _apply),
    (["salary", "pay", "ctc", "stipend", "money", "compensation"], lambda c: _field(c, "Salary")),
    (["location", "where", "city", "remote", "office", "hybrid"], lambda c: _field(c, "Location")),
    (["shift", "timing", "working hours"], lambda c: _field(c, "Shift")),
    (["experience", "fresher", "year*"], lambda c: _field(c, "Experience")),
    (["review*", "positive", "concern", "culture", "employer", "pros", "cons", "company*"], _company),
    (["actually do", "what will i do", "responsibilit*", "day to day", "work on", "role involve"], _do),
    (["skill*", "missing", "gap*", "lack*", "learn*"], _skills),
    (["suit*", "fit", "match", "score", "good for me"], _fit),
    (["help", "hello", "hi", "what can you do"], _help),
]


def _llm_fallback(question, ctx):
    job, fit = ctx["job"], ctx["fit"]
    facts = {"job": {k: v for k, v in job.items() if k != "raw_text"},
             "fit_percent": fit["percent"], "fit_notes": [c["note"] for c in fit["checks"]],
             "company": ctx.get("company"), "reviews": ctx.get("reviews")}
    reply = ask_llm("You are JobLens. Answer the question using ONLY the CONTEXT below. "
                    f"If the answer is not in the context, reply exactly: {NO_INFO}\n\n"
                    f"CONTEXT:\n{json.dumps(facts, default=str, ensure_ascii=False)}\n\nQUESTION: {question}")
    return reply or NO_INFO


def answer(question, ctx):
    """ctx = {'job', 'fit', 'plain', 'company', 'reviews'} from the last analysis."""
    if not ctx or not ctx.get("job"):
        return "Please analyze a job first, then ask me about it."
    if not isinstance(question, str) or not question.strip():
        return "Please enter a question about the analyzed job."
    q = _normalize(question)
    for keywords, handler in INTENTS:
        if any(_matches(q, k) for k in keywords):
            return handler(ctx)
    return _llm_fallback(question.strip(), ctx)
