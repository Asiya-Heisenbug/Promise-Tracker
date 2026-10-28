"""
scraper.py - tiny, polite web helpers.

Everything here fails gracefully: if a site blocks us (robots.txt, 403, JavaScript-only
pages) we return a short message and the app falls back to text the user pastes.
There are deliberately NO workarounds, and no scraping of review sites (their terms
usually forbid it); reviews are pasted by the user instead.
"""
import json
import re
from urllib.parse import quote, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from nlp import extract_location

HEADERS = {"User-Agent": "JobLens-College-Prototype/0.1 (educational project)"}


def _allowed(url):
    """Respect robots.txt. A missing robots.txt means allowed."""
    p = urlparse(url)
    try:
        r = requests.get(f"{p.scheme}://{p.netloc}/robots.txt", headers=HEADERS, timeout=5)
    except requests.RequestException:
        return True  # if the site is unreachable the real request fails with a clear message
    if r.status_code != 200:
        return True
    rp = RobotFileParser()
    rp.parse(r.text.splitlines())
    return rp.can_fetch(HEADERS["User-Agent"], url)


def _jobposting_text(soup):
    """Many job pages embed a schema.org JobPosting as JSON-LD: much cleaner than page text."""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except ValueError:
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data])
        for it in items:
            if not (isinstance(it, dict) and it.get("@type") == "JobPosting"):
                continue
            loc = it.get("jobLocation")
            loc = loc[0] if isinstance(loc, list) and loc else loc
            city = ((loc or {}).get("address") or {}).get("addressLocality", "") if isinstance(loc, dict) else ""
            sal = (it.get("baseSalary") or {}).get("value") or {}
            sal_line = ""
            if isinstance(sal, dict) and sal.get("minValue"):
                cur = (it.get("baseSalary") or {}).get("currency", "INR")
                sal_line = f"Salary: {cur} {sal['minValue']} - {sal.get('maxValue', sal['minValue'])} per {str(sal.get('unitText', '')).lower()}"
            desc = BeautifulSoup(it.get("description", ""), "html.parser").get_text("\n")
            org = (it.get("hiringOrganization") or {}).get("name", "")
            return "\n".join(x for x in [it.get("title", ""), f"Company: {org}" if org else "",
                                         f"Location: {city}" if city else "", sal_line, desc] if x)
    return None


def fetch_page_text(url):
    """Returns (text, error_message). Never raises."""
    url = (url or "").strip()
    if not url:
        return None, "Please enter a URL first."
    if not url.startswith("http"):
        url = "https://" + url
    try:
        if not _allowed(url):
            return None, "This site's robots.txt doesn't allow automated access. Please paste the text instead."
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None, f"The site returned status {r.status_code} (it may block scrapers). Please paste the text instead."
        soup = BeautifulSoup(r.text, "html.parser")
        text = _jobposting_text(soup)
        if not text:  # no structured data: fall back to the visible page text
            for t in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                t.decompose()
            text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n")).strip()
            if len(text) < 200:
                return None, "The page had almost no readable text (it probably loads content with JavaScript). Please paste the text instead."
        return text[:8000], None
    except Exception as e:
        return None, f"Couldn't fetch the page ({type(e).__name__}). Please paste the text instead."


# ----------------------------------------------------------------------------
# Company information (very small on purpose)
# ----------------------------------------------------------------------------
_INDUSTRIES = [
    (r"\bai\b|artificial intelligence|machine learning|software|saas|it services|technology", "Technology / Software"),
    (r"\bbank|insurance|fintech|payments|financial", "Banking / Finance"),
    (r"e-?commerce|retail", "E-commerce / Retail"),
    (r"health|hospital|pharma|medical", "Healthcare"),
    (r"education|edtech", "Education"),
    (r"manufactur|automotive", "Manufacturing"),
    (r"consulting", "Consulting"),
    (r"telecom", "Telecom"),
]


def _industry(text):
    t = text.lower()
    return next((name for pat, name in _INDUSTRIES if re.search(pat, t)), None)


def _site_description(url):
    """The <meta name="description"> of a company website, if robots.txt allows."""
    url = url if url.startswith("http") else "https://" + url
    if not _allowed(url):
        return None
    r = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code != 200:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    return meta["content"].strip() if meta and meta.get("content") else None


def _wikipedia(name):
    """Search Wikipedia for a likely company page; return (extract, page_url) or None."""
    ignored = {"a", "an", "and", "company", "corp", "corporation", "inc", "limited",
               "llc", "ltd", "of", "plc", "private", "pvt", "the"}
    wanted = {word for word in re.findall(r"[a-z0-9]+", name.lower()) if word not in ignored}
    if not wanted:
        return None

    results = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": name, "srlimit": 5, "format": "json"},
        headers=HEADERS,
        timeout=8,
    )
    if results.status_code != 200:
        return None

    candidates = results.json().get("query", {}).get("search", [])
    ranked = []
    for result in candidates:
        title = result.get("title", "")
        words = {word for word in re.findall(r"[a-z0-9]+", title.lower()) if word not in ignored}
        overlap = wanted & words
        if overlap and len(overlap) / len(wanted) >= 0.5:
            ranked.append((len(overlap) / len(wanted), title))

    for _, title in sorted(ranked, reverse=True):
        response = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title.replace(" ", "_")),
            headers=HEADERS,
            timeout=8,
        )
        if response.status_code != 200:
            continue
        data = response.json()
        if data.get("type") == "standard" and data.get("extract"):
            return data["extract"], data["content_urls"]["desktop"]["page"]
    return None


def get_company_info(name, website="", pasted="", use_wiki=True):
    """Returns {'description', 'industry', 'location', 'source'} or None.
    Priority: text the user pasted > the company website's meta description > Wikipedia search."""
    desc = source = None
    try:
        if (pasted or "").strip():
            desc, source = pasted.strip(), "Pasted by you"
        elif (website or "").strip():
            desc = _site_description(website.strip())
            source = website.strip() if desc else None
        if not desc and use_wiki and name:
            found = _wikipedia(name)
            if found:
                desc, source = found
    except Exception:
        desc = None  # offline / blocked: the app just shows the "paste it yourself" hint
    if not desc:
        return None
    return {"description": desc[:700], "industry": _industry(desc), "location": extract_location(desc), "source": source}
