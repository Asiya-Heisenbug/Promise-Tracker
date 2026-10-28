"""
nlp.py - the NLP core of JobLens.

Turns raw job-description text into structured fields using simple, explainable NLP:
  * regex                              -> salary, experience, work mode, shift, location
  * keyword vocabulary + aliases       -> skills ("ML" -> "machine learning")
  * section splitting                  -> required vs preferred skills, responsibilities
  * spaCy NER (optional)               -> company / location fallback, entity display
  * sentence-transformers (optional)   -> semantic similarity (used by analyzer.py)
  * Gemini API (optional)              -> only for rewriting / chat, never for extraction

Optional libraries are imported defensively: if one is missing, the app still runs
with a simpler fallback.
"""
import importlib.util
import os
import re
from functools import lru_cache
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

# ----------------------------------------------------------------------------
# Optional libraries
# ----------------------------------------------------------------------------
try:
    import spacy

    _spacy = spacy.load("en_core_web_sm")
except Exception:  # spaCy or its model is not installed
    _spacy = None

_embedder = None  # sentence-transformers model, loaded on first use


def get_embedder():
    """Load the sentence-transformers model once. Returns None if unavailable."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _embedder = False
    return _embedder or None


@lru_cache(maxsize=512)
def _embed(text):
    return get_embedder().encode(text, convert_to_tensor=True)


_ROLE_SYNONYMS = {"developer": "engineer", "programmer": "engineer",
                  "artificial intelligence": "ai", "machine learning": "ml"}
_FILLER = {"a", "an", "the", "of", "and", "fresher", "junior", "senior", "intern", "trainee", "entry", "level"}


def _tokens(text):
    t = text.lower()
    for k, v in _ROLE_SYNONYMS.items():
        t = t.replace(k, v)
    return {w for w in re.findall(r"[a-z0-9+#]+", t) if w not in _FILLER}


def similarity(a, b):
    """Semantic similarity (0..1) between two short texts.
    Uses sentence embeddings when available, otherwise simple word overlap."""
    if get_embedder():
        from sentence_transformers import util

        return max(0.0, float(util.cos_sim(_embed(a), _embed(b))))
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def ask_llm(prompt):
    """Call Gemini if GEMINI_API_KEY is set. Returns text, or None on any problem."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # change via env var if needed
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        r = requests.post(url, headers={"x-goog-api-key": key}, timeout=30,
                          json={"contents": [{"parts": [{"text": prompt}]}]})
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return None


def engine_status():
    """Which NLP engines are active (shown in the sidebar)."""
    return {
        "spaCy (entities)": _spacy is not None,
        "sentence-transformers (semantic match)": importlib.util.find_spec("sentence_transformers") is not None,
        "Gemini (optional rewrite / chat)": bool(os.getenv("GEMINI_API_KEY")),
    }


def entities(text):
    """Named entities found by spaCy (empty list if spaCy is not installed)."""
    if not _spacy:
        return []
    doc = _spacy(text[:3000])
    wanted = {"ORG", "GPE", "MONEY", "DATE"}
    found = {(e.text.strip(), e.label_) for e in doc.ents if e.label_ in wanted}
    return sorted((t, l) for t, l in found if len(t) > 1 and "\n" not in t)  # drop multi-line fragments


# ----------------------------------------------------------------------------
# Skills: vocabulary with aliases
# ----------------------------------------------------------------------------
SKILLS = {  # canonical name -> aliases (lowercase)
    "python": ["python"], "java": ["java"], "javascript": ["javascript", "js"], "c++": ["c++"],
    "sql": ["sql", "mysql", "postgresql", "postgres"], "mongodb": ["mongodb"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "neural network", "neural networks"],
    "nlp": ["nlp", "natural language processing"],
    "llm": ["llm", "llms", "large language model", "large language models"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "generative ai": ["generative ai", "genai", "gen ai"],
    "computer vision": ["computer vision", "opencv"],
    "fastapi": ["fastapi"], "flask": ["flask"], "django": ["django"],
    "rest apis": ["rest api", "rest apis", "restful", "api", "apis"],
    "pandas": ["pandas"], "numpy": ["numpy"], "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "tensorflow": ["tensorflow"], "pytorch": ["pytorch"], "spacy": ["spacy"],
    "hugging face": ["hugging face", "huggingface", "transformers"], "langchain": ["langchain"],
    "docker": ["docker"], "kubernetes": ["kubernetes", "k8s"], "git": ["git", "github"],
    "aws": ["aws", "amazon web services"], "azure": ["azure"], "gcp": ["gcp", "google cloud"],
    "linux": ["linux"], "mlops": ["mlops"], "data analysis": ["data analysis", "data analytics"],
    "react": ["react", "reactjs"], "power bi": ["power bi", "powerbi"], "tableau": ["tableau"],
    "excel": ["ms excel", "microsoft excel", "advanced excel"],
}
_ALIAS_TO_SKILL = {a: s for s, aliases in SKILLS.items() for a in aliases}
_SKILL_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(re.escape(a) for a in sorted(_ALIAS_TO_SKILL, key=len, reverse=True))
    + r")(?![A-Za-z0-9+#])", re.I)


def find_skills(text):
    """Skills mentioned in text, as canonical names, in order of first appearance."""
    return list(dict.fromkeys(_ALIAS_TO_SKILL[m.group(1).lower()] for m in _SKILL_RE.finditer(text)))


def normalize_skill(skill):
    """'ML' -> 'machine learning'. Unknown skills are just lower-cased."""
    s = skill.strip().lower()
    return _ALIAS_TO_SKILL.get(s, s)


# ----------------------------------------------------------------------------
# Regex extractors
# ----------------------------------------------------------------------------
_NUM = r"(\d[\d,]*(?:\.\d+)?)"
_CUR = r"(?:₹|rs\.?|inr)"
_SAL_A = re.compile(  # ₹40,000 - ₹60,000   |   Rs. 40k   |   INR 3 - 4.5 lakh
    rf"(?<![A-Za-z]){_CUR}\s*{_NUM}(?:\s*(k|lakhs?|lpa)\b)?"
    rf"(?:\s*(?:-|–|—|to)\s*(?:{_CUR}\s*)?{_NUM}(?:\s*(k|lakhs?|lpa)\b)?)?", re.I)
_SAL_B = re.compile(rf"{_NUM}(?:\s*(?:-|–|—|to)\s*{_NUM})?\s*(lpa|lakhs?)\b", re.I)  # 4-6 LPA


def _rupees(num, unit):
    n = float(num.replace(",", ""))
    unit = (unit or "").lower()
    return n * (1000 if unit == "k" else 100000 if unit in ("lpa", "lakh", "lakhs") else 1)


def extract_salary(text):
    """Returns {'text', 'min_monthly', 'max_monthly'} (INR per month) or None."""
    for pat in (_SAL_A, _SAL_B):
        m = pat.search(text)
        if not m:
            continue
        if pat is _SAL_A:
            n1, u1, n2, u2 = m.groups()
        else:
            n1, n2, u1 = m.groups()
            u2 = None
        unit = u1 or u2
        lo = _rupees(n1, unit)
        hi = _rupees(n2, unit) if n2 else lo
        lo, hi = min(lo, hi), max(lo, hi)
        tail = text[m.end():]
        period = re.match(r"\s*(?:(?:per|/|a)\s*(?:month|annum|year|yr|mo)\b|(?:pm|pa|monthly|annually|yearly)\b)", tail, re.I)
        after = tail[:40].lower()
        if re.search(r"month|/\s*mo\b|\bpm\b", after):
            div = 1
        elif (unit or "").lower() in ("lpa", "lakh", "lakhs") or re.search(r"annum|annual|year|\bpa\b|ctc|/\s*yr", after) or hi >= 200000:
            div = 12
        else:
            div = 1
        shown = text[m.start(): m.end() + (period.end() if period else 0)].strip()
        return {"text": shown, "min_monthly": round(lo / div), "max_monthly": round(hi / div)}
    return None


def extract_experience(text):
    """Returns {'text', 'min_years'} or None."""
    t = text.lower()
    m = re.search(r"(\d+)\s*(?:-|–|to)\s*(\d+)\s*\+?\s*(?:years?|yrs?)", t)
    if m:
        return {"text": f"{m.group(1)}-{m.group(2)} years", "min_years": int(m.group(1))}
    m = re.search(r"(\d+)\s*\+\s*(?:years?|yrs?)|(?:minimum|at least)\s*(\d+)\s*(?:years?|yrs?)"
                  r"|(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:relevant\s*)?experience", t)
    if m:
        n = int(next(g for g in m.groups() if g))
        return {"text": f"{n}+ years", "min_years": n}
    if re.search(r"\bfreshers?\b|entry[- ]level|\b0\s*years", t):
        return {"text": "Fresher / entry level", "min_years": 0}
    return None


_CITY_RE = re.compile(r"\b(Chennai|Bengaluru|Bangalore|Hyderabad|Mumbai|Pune|New Delhi|Delhi|Gurugram|Gurgaon|"
                      r"Noida|Kolkata|Coimbatore|Kochi|Ahmedabad|Madurai|Trichy)\b", re.I)


def extract_location(text):
    m = re.search(r"(?im)^\s*(?:job\s+)?locations?\s*[:\-]\s*(.+)$", text)
    if m:
        return m.group(1).strip()
    m = _CITY_RE.search(text)
    if m:
        return m.group(1).title()
    if _spacy:
        for ent in _spacy(text[:2000]).ents:
            if ent.label_ == "GPE":
                return ent.text
    return None


def extract_work_mode(text):
    t = text.lower()
    if re.search(r"\bhybrid\b", t):
        return "Hybrid"
    if re.search(r"\bremote\b|work from home|\bwfh\b", t):
        return "Remote"
    if re.search(r"on-?site|work from office|\bwfo\b|in-office", t):
        return "On-site"
    return None


_SHIFT_PATTERNS = [  # categories shared with the user's shift preference
    ("Night", r"night shift|\bus shift|us timings|uk shift|graveyard"),
    ("Rotational", r"rotational|rotating shift"),
    ("Flexible", r"flexible (?:hours|timings?|shifts?)"),
    ("Evening", r"evening shift"),
    ("Morning/Day", r"morning shift|day shift|general shift|\b(?:8|9|10)(?::\d\d)?\s*(?:am)?\s*(?:-|–|to)\s*(?:5|6|7)(?::\d\d)?\s*(?:pm)?\b"),
]


def extract_shift(text):
    t = text.lower()
    for name, pat in _SHIFT_PATTERNS:
        if re.search(pat, t):
            return name
    return None


def extract_title(text):
    m = re.search(r"(?im)^\s*#{0,6}\s*(?:job\s*title|position|designation|role)\s*[:\-]\s*(.+)$", text)
    if m:
        return m.group(1).strip().lstrip("#").strip()
    for line in text.splitlines():
        line = re.sub(r"^\s*#{1,6}\s*", "", line).strip()
        description = re.match(r"(?i)^job description\s*[:\-]\s*(.+)$", line)
        if description:
            return description.group(1).strip()
        if line and not re.match(r"(?i)^job description\b", line):
            looks_like_title = len(line) <= 80 and len(line.split()) <= 10 and not line.endswith((".", "!", "?"))
            return line if looks_like_title else None
    return None


def extract_company(text):
    title = extract_title(text)

    def is_job_title(value):
        normalized = re.sub(r"\W+", " ", value).strip().casefold()
        title_normalized = re.sub(r"\W+", " ", title or "").strip().casefold()
        return bool(normalized and title_normalized and (
            normalized == title_normalized
            or normalized in title_normalized
            or title_normalized in normalized
        ))

    m = re.search(r"(?im)^\s*company(?:\s*name)?\s*[:\-]\s*(.+)$", text)
    if m:
        company = m.group(1).strip()
        return None if is_job_title(company) else company
    m = re.search(r"(?m)^\s*[Aa]bout\s+(?!(?:us|the|you|this|role|team)\b)([A-Z][^\n:]{1,40}?)\s*:?\s*$", text)
    if m:
        company = m.group(1).strip()
        return None if is_job_title(company) else company
    m = re.search(r"\b[Jj]oin\s+([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,3})", text)
    if m:
        return m.group(1).strip(" .,")
    if _spacy:  # fallback: first organisation spaCy finds that isn't a tech skill
        for ent in _spacy(text[:1500]).ents:
            if ent.label_ == "ORG" and ent.text.lower() not in _ALIAS_TO_SKILL and not is_job_title(ent.text):
                return ent.text
    return None


# ----------------------------------------------------------------------------
# Sections, responsibilities and the main entry point
# ----------------------------------------------------------------------------
_HEADERS = [  # first match wins, so "Preferred skills" is not mistaken for "skills"
    ("preferred", r"preferred|nice.to.have|good.to.have|bonus|desirable|added advantage|plus"),
    ("responsibilities", r"responsibilit|what you.{0,3}ll do|what you will do|your role|duties|key tasks|day.to.day"),
    ("required", r"requirement|required|must.have|qualification|looking for|what you.{0,3}ll need|eligibility|skills"),
    ("benefits", r"benefit|perks|what we offer"),
    ("about", r"^about|who we are|overview"),
]
_SMALL = {"to", "and", "of", "the", "you", "we", "will", "what", "do", "have", "for", "you'll", "you’ll",
          "who", "are", "a", "in", "your", "we're", "we’re"}


def split_sections(text):
    """Group lines under headings like 'Responsibilities', 'Requirements', 'Good to have'."""
    sections = {k: [] for k in ("responsibilities", "required", "preferred", "benefits", "about", "other")}
    current = "other"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        head, colon, rest = line.partition(":")
        words = head.split()
        if colon:
            is_header = 0 < len(words) <= 6
        else:
            is_header = (0 < len(words) <= 4 and not re.match(r"^[-•*·●▪\d]", line)
                         and all(w[0].isupper() or w.lower() in _SMALL for w in words))
        name = next((n for n, pat in _HEADERS if is_header and re.search(pat, head.lower())), None)
        if name and rest.strip():
            sections[name].append(rest.strip())   # "Required Skills: Python, SQL" (one line)
        elif name:
            current = name                        # a heading on its own line
        else:
            sections[current].append(line)
    return sections


_VERBS = (r"develop|build|design|implement|work|collaborate|maintain|deploy|analy[sz]e|write|create|"
          r"integrate|train|test|lead|manage|support|research|optimi[sz]e|leverage|clean|evaluate|"
          r"generate|prospect|negotiate|sell|present|achieve|identify|coordinate|prepare|monitor|communicate")


def extract_responsibilities(sections, text):
    lines = [re.sub(r"^[-•*·●▪\d.)\s]+", "", l).strip() for l in sections["responsibilities"]]
    lines = [l for l in lines if len(l) > 15]
    if not lines:  # no heading found: pick sentences that start with an action verb
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        lines = [s.strip() for s in sentences if len(s) > 25 and re.match(rf"(?i)\s*(?:{_VERBS})\w*\b", s)]
    return lines[:8]


def extract_job(text):
    """Main entry point: raw job description -> dict of structured fields."""
    sec = split_sections(text)
    required = find_skills(" ".join(sec["required"]))
    preferred = [s for s in find_skills(" ".join(sec["preferred"])) if s not in required]
    loose = find_skills(" ".join(sec["other"] + sec["responsibilities"]))   # skills mentioned elsewhere
    if not required:  # no explicit requirements section: fall back to skills mentioned anywhere
        required = [s for s in loose if s not in preferred]
    return {
        "title": extract_title(text),
        "company": extract_company(text),
        "location": extract_location(text),
        "salary": extract_salary(text),
        "experience": extract_experience(text),
        "work_mode": extract_work_mode(text),
        "shift": extract_shift(text),
        "required_skills": required,
        "preferred_skills": preferred,
        "responsibilities": extract_responsibilities(sec, text),
        "raw_text": text,
    }
