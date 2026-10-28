"""
analyzer.py - compares a job with the user's preferences and explains it.

  fit_score()        weighted comparison of preferences vs job  (an ESTIMATE, not science)
  plain_language()   rewrites corporate job-description wording into simple English
  analyze_reviews()  finds repeated themes in a few pasted employee reviews
"""
import re

from nlp import ask_llm, get_embedder, normalize_skill, similarity

# How much each criterion counts. Criteria the job doesn't mention are skipped and
# the remaining weights are re-scaled, so missing data never silently counts as a "no".
WEIGHTS = {"Skills": 30, "Role": 20, "Experience": 15, "Salary": 15, "Location": 10, "Shift": 10}


def _city(s):
    return s.lower().replace("bangalore", "bengaluru").replace("gurgaon", "gurugram")


def _rs(n):
    return f"₹{n:,}"


# ----------------------------------------------------------------------------
# One small function per criterion -> (score 0..1 or None, explanation)
# ----------------------------------------------------------------------------
def check_role(prefs, job):
    if not job["title"] or not prefs["role"].strip():
        return None, "The job title or your preferred role is missing."
    sim = similarity(prefs["role"], job["title"])
    score = min(1.0, sim / 0.6)
    if score >= 0.8:
        return score, f"The job title “{job['title']}” is close to your preferred role “{prefs['role']}”."
    if score >= 0.4:
        return score, f"The job title “{job['title']}” is only loosely related to “{prefs['role']}”."
    return score, f"The job title “{job['title']}” looks quite different from “{prefs['role']}”."


def check_location(prefs, job):
    loc, want = job["location"], prefs["location"].strip()
    if job["work_mode"] == "Remote":
        return 1.0, "The role is remote, so location doesn't matter."
    if not loc or not want:
        return None, "Location is not clearly mentioned."
    if _city(want) in _city(loc):
        return 1.0, f"The job is in {loc}, matching your preferred location ({want})."
    return 0.0, f"The job is in {loc}, not {want}."


def check_salary(prefs, job):
    sal, need = job["salary"], prefs["min_salary"]
    if not sal:
        return None, "The job description does not mention salary."
    if not need:
        return None, "You didn't set a minimum salary."
    lo, hi = sal["min_monthly"], sal["max_monthly"]
    rng = _rs(lo) if lo == hi else f"{_rs(lo)}–{_rs(hi)}"
    if lo >= need:
        return 1.0, f"The listed salary ({rng}/month) meets your minimum of {_rs(need)}."
    if hi >= need:
        return 0.75, f"The salary range ({rng}/month) reaches your minimum of {_rs(need)} only at the upper end."
    return max(0.0, 1 - 2 * (need - hi) / need), f"The listed salary ({rng}/month) is below your minimum of {_rs(need)}."


def check_shift(prefs, job):
    want = {"Morning": "Morning/Day", "Evening": "Evening", "Night": "Night"}.get(prefs["shift"])
    if want is None:
        return None, "You have no shift preference."
    if not job["shift"]:
        return None, "The shift is not mentioned."
    if job["shift"] in (want, "Flexible"):
        return 1.0, f"The shift ({job['shift']}) fits your preference."
    if job["shift"] == "Rotational":
        return 0.5, "The job has rotational shifts, so it may not always match your preferred shift."
    return 0.0, f"The job is a {job['shift']} shift, but you prefer {prefs['shift'].lower()}."


def check_experience(prefs, job):
    exp = job["experience"]
    if not exp:
        return None, "Required experience is not mentioned."
    t = prefs["experience"].lower()
    m = re.search(r"\d+(?:\.\d+)?", t)
    have = 0.0 if ("fresher" in t or not m) else float(m.group())
    need = exp["min_years"]
    if have >= need:
        return 1.0, f"The job asks for {exp['text']}, which you meet."
    return max(0.0, 1 - (need - have) / 2), f"The job asks for {exp['text']}, a bit more than your {prefs['experience']}."


def check_skills(prefs, job):
    mine = {normalize_skill(s) for s in re.split(r"[,;/\n]+", prefs["skills"]) if s.strip()}
    req, pref = job["required_skills"], job["preferred_skills"]
    info = {"mine": mine, "matched": [], "related": {}, "missing_required": [], "missing_preferred": []}
    if not (req or pref):
        return None, "No specific skills were found in the job description.", info

    def credit(skill):
        if skill in mine:
            return 1.0
        if mine and get_embedder():  # semantic match: e.g. PyTorch ~ TensorFlow
            best, sim = max(((m, similarity(skill, m)) for m in mine), key=lambda x: x[1])
            if sim >= 0.7:
                info["related"][skill] = best
                return 0.5
        return 0.0

    def coverage(skills):
        return sum(credit(s) for s in skills) / len(skills) if skills else None

    rc, pc = coverage(req), coverage(pref)
    score = rc * 0.75 + pc * 0.25 if (rc is not None and pc is not None) else (rc if rc is not None else pc)
    info["matched"] = [s for s in req + pref if s in mine]
    info["missing_required"] = [s for s in req if s not in mine and s not in info["related"]]
    info["missing_preferred"] = [s for s in pref if s not in mine and s not in info["related"]]
    note = f"You list {len(info['matched'])} of the {len(req) + len(pref)} skills mentioned."
    if info["missing_required"]:
        note += f" Missing required: {', '.join(info['missing_required'])}."
    if info["missing_preferred"]:
        note += f" Nice-to-have you don't list: {', '.join(info['missing_preferred'])}."
    return score, note, info


# ----------------------------------------------------------------------------
# Fit score
# ----------------------------------------------------------------------------
def fit_score(prefs, job):
    """Returns {'percent', 'label', 'explanation', 'checks', 'skills'}."""
    skill_score, skill_note, skill_info = check_skills(prefs, job)
    results = {
        "Role": check_role(prefs, job),
        "Location": check_location(prefs, job),
        "Salary": check_salary(prefs, job),
        "Shift": check_shift(prefs, job),
        "Experience": check_experience(prefs, job),
        "Skills": (skill_score, skill_note),
    }
    checks = [{"name": n, "score": s, "note": note, "weight": WEIGHTS[n]} for n, (s, note) in results.items()]
    known = [c for c in checks if c["score"] is not None]
    total = sum(c["weight"] for c in known)
    # with fewer than 3 usable criteria a percentage would be misleading, so we don't show one
    percent = round(100 * sum(c["score"] * c["weight"] for c in known) / total) if len(known) >= 3 else None

    good = [c["note"] for c in known if c["score"] >= 0.8]
    weak = [c["note"] for c in known if c["score"] < 0.8]
    unknown = [c["name"].lower() for c in checks if c["score"] is None]
    parts = []
    if good:
        parts.append("**Why it fits:** " + " ".join(good))
    if weak:
        parts.append("**Main gaps:** " + " ".join(weak))
    if unknown:
        parts.append(f"**Not enough information to judge:** {', '.join(unknown)}.")
    label = None if percent is None else "Strong match" if percent >= 75 else "Moderate match" if percent >= 50 else "Weak match"
    return {"percent": percent, "label": label, "explanation": "\n\n".join(parts),
            "checks": checks, "skills": skill_info, "criteria_used": f"{len(known)} of {len(checks)}"}


# ----------------------------------------------------------------------------
# "What will I actually do?"
# ----------------------------------------------------------------------------
JARGON = {  # corporate phrase -> plain English
    "intelligent solutions": "smart features", "cross-functional stakeholders": "people from other teams and clients",
    "cross-functional": "multi-team", "stakeholders": "teams and clients", "end-to-end": "full",
    "best practices": "good engineering habits", "state-of-the-art": "latest", "cutting-edge": "modern",
    "collaborate with": "work with", "fast-paced": "busy", "mission-critical": "very important",
    "operationalize": "put into real use", "leverage": "use", "utilize": "use", "facilitate": "help with",
    "spearhead": "lead", "ideate": "brainstorm", "scalable": "easy to grow", "robust": "reliable",
    "seamless": "smooth", "deliverables": "work output", "ecosystem": "set of tools",
    "proactively": "on your own initiative", "synergy": "teamwork", "dynamic": "changing",
}
ACTIVITIES = [  # (pattern, plain description)
    (r"\bsales?\b|revenue|quota|\bsales targets?\b", "meeting sales targets and growing revenue"),
    (r"lead generation|generat(?:e|ing) leads?|prospect(?:ing)?|qualif(?:y|ying) leads",
     "finding and qualifying potential customers"),
    (r"customer relationships?|client relationships?|account management", "building and managing customer relationships"),
    (r"negotiat|close deals|closing deals|sales presentation", "presenting offers, negotiating and closing deals"),
    (r"\bcrm\b|salesforce|hubspot|sales pipeline", "updating customer records and tracking the sales pipeline"),
    (r"\bmodels?\b|machine learning|\bml\b|\btrain", "building and evaluating machine-learning models"),
    (r"\bnlp\b|natural language|text data|\btext\b", "working with text data (NLP)"),
    (r"\bllms?\b|\brag\b|prompt|generative", "building features on top of large language models"),
    (r"\bapis?\b|fastapi|flask|backend|\brest\b", "writing backend APIs so apps can use the models"),
    (r"deploy|production|docker|cloud|\baws\b|azure", "deploying work to the cloud"),
    (r"dataset|data (?:cleaning|pipeline|preprocess|analysis)|\betl\b|preprocess", "preparing and analysing datasets"),
    (r"\btest|debug|code review", "testing, debugging and code reviews"),
    (r"collaborat|stakeholder|cross-functional|\bclients?\b", "coordinating with teammates and clients"),
    (r"document|\breports?\b|present", "writing documentation and reports"),
    (r"front-?end|\breact\b|\bui\b", "building parts of the user interface"),
]


def _simplify(sentence):
    for phrase in sorted(JARGON, key=len, reverse=True):
        sentence = re.sub(rf"\b{re.escape(phrase)}\b", JARGON[phrase], sentence, flags=re.I)
    return sentence[:1].upper() + sentence[1:]


def plain_language(job):
    """Returns {'summary', 'bullets', 'translations', 'by_llm'}."""
    resp = job["responsibilities"]
    text = " ".join(resp).lower()
    activities = [desc for pat, desc in ACTIVITIES if re.search(pat, text)]
    if activities:
        top = activities[:3]
        summary = "You'll mainly spend your time " + (", ".join(top[:-1]) + " and " + top[-1] if len(top) > 1 else top[0]) + "."
    else:
        summary = "I couldn't find enough information to describe the day-to-day work."
    by_llm = False
    if resp:  # optional: let an LLM rewrite it, grounded in the extracted responsibilities only
        llm = ask_llm("Rewrite these job responsibilities as 2-3 plain-English sentences starting with "
                      "\"You'll mainly\". Use ONLY the information given; do not add anything.\n\n- " + "\n- ".join(resp))
        if llm:
            summary, by_llm = llm, True
    return {"summary": summary, "bullets": [a[:1].upper() + a[1:] for a in activities],
            "translations": [(r, _simplify(r)) for r in resp[:6]], "by_llm": by_llm}


# ----------------------------------------------------------------------------
# Employee review themes (simple keyword + sentiment-word analysis)
# ----------------------------------------------------------------------------
THEMES = {  # theme -> keywords (single words match word prefixes, phrases match as text)
    "Learning": ["learn", "mentor", "training", "skill"],
    "Growth": ["growth", "grow", "promot", "career", "opportunit"],
    "Work culture": ["culture", "friendly", "colleague", "team", "environment", "supportive", "seniors"],
    "Management": ["manager", "management", "leadership", "hr"],
    "Salary": ["salary", "pay", "hike", "package", "increment", "compensation"],
    "Work-life balance": ["work life", "work-life", "balance", "flexible", "timings", "wfh", "work from home"],
    "Workload": ["workload", "pressure", "stress", "deadline", "burnout", "hectic"],
    "Overtime": ["overtime", "long hours", "late night", "extra hours", "weekend"],
}
POSITIVE = {"good", "great", "excellent", "amazing", "supportive", "friendly", "helpful", "best", "nice", "positive",
            "flexible", "fair", "learn", "learned", "learning", "growth", "opportunity", "opportunities", "balanced",
            "collaborative", "transparent", "respectful", "approachable", "understanding", "encouraging",
            "pleasant", "comfortable", "relaxed"}
NEGATIVE = {"bad", "poor", "toxic", "low", "underpaid", "stress", "stressful", "pressure", "overtime", "long",
            "micromanagement", "delay", "delayed", "politics", "burnout", "slow", "unfair", "worst", "heavy",
            "hectic", "excessive", "overloaded", "unprofessional"}
NEGATORS = {"no", "not", "never", "without", "hardly", "barely", "lack", "lacking", "lacks", "isn't", "aren't",
            "don't", "doesn't", "didn't", "wasn't", "cannot", "can't"}
NEGATIVE_PHRASES = ["no work life", "no work-life", "no hike", "no increment", "no appreciation", "no flexibility"]


def _polarity(clause, tokens):
    """+ for positive wording, - for negative. A negator in the 3 words before flips the sign."""
    score = -sum(clause.count(p) for p in NEGATIVE_PHRASES)
    for i, tok in enumerate(tokens):
        sign = 1 if tok in POSITIVE else -1 if tok in NEGATIVE else 0
        if sign and any(w in NEGATORS for w in tokens[max(0, i - 3):i]):
            sign = -sign
        score += sign
    if {"rare", "rarely", "seldom"} & set(tokens):  # "overtime is rare" -> good news
        score = -score if score else 1
    return score


def _mentions(clause, tokens, keywords):
    for k in keywords:
        if " " in k or "-" in k:
            if k in clause:
                return True
        elif any(t.startswith(k) for t in tokens):
            return True
    return False


def analyze_reviews(text, source=""):
    """Returns {'n', 'source', 'positive': [...], 'negative': [...]} or None if there are no reviews.
    Each theme entry is {'theme', 'count' (reviews mentioning it), 'example' (a short clause)}."""
    reviews = [r.strip() for r in re.split(r"\n\s*\n", text or "") if r.strip()]
    if len(reviews) == 1:  # maybe one review per line instead
        reviews = [l.strip(" -•*\t") for l in reviews[0].splitlines() if l.strip()]
    reviews = reviews[:30]
    if not reviews:
        return None
    tally = {"positive": {}, "negative": {}}
    for review in reviews:
        score, example = {}, {}
        # split into clauses so "Learning is good but the workload is heavy" is judged per part
        for clause in re.split(r"[.!?;\n]+|\b(?:but|though|however|although|while|yet)\b", review, flags=re.I):
            clause = clause.strip()
            low = clause.lower()
            tokens = re.findall(r"[a-z']+", low)
            pol = _polarity(low, tokens) if len(clause) >= 5 else 0
            if pol == 0:
                continue
            for theme, kws in THEMES.items():
                if _mentions(low, tokens, kws):
                    score[theme] = score.get(theme, 0) + pol
                    example.setdefault((theme, pol > 0), clause[:120])
        for theme, s in score.items():
            if s != 0:
                side = tally["positive" if s > 0 else "negative"].setdefault(
                    theme, {"theme": theme, "count": 0, "example": example.get((theme, s > 0), "")})
                side["count"] += 1
    ranked = lambda d: sorted(d.values(), key=lambda t: -t["count"])  # noqa: E731
    return {"n": len(reviews), "source": source.strip(), "positive": ranked(tally["positive"]),
            "negative": ranked(tally["negative"])}
