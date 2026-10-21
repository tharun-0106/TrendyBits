"""Dependency-free NLP helpers shared by every module.

Tokenisation, sentence splitting, light stemming, named-entity / quantity extraction,
hedge detection, topic taxonomy and the concept lexicon used by the offline
semantic-embedding fallback.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable

STOPWORDS = set(
    """a an the and or but if then else of to in on at by for with from as is are was were be been being
    this that these those it its he she they them his her their we our you your i me my so than too very
    can will just do does did done has have had having into over under about after before between during
    through up down out off again further once here there when where why how both each few more most other
    such own same s t should would shall also said says say per via while which who whom whose what
    am being get got let us one two""".split()
)
NEGATIONS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "cannot", "without",
             "denied", "denies", "deny", "isn't", "wasn't", "aren't", "doesn't", "didn't", "hasn't",
             "haven't", "won't", "don't", "can't"}
DEBUNK_CUES = ["fake", "false", "hoax", "baseless", "fabricated", "debunk", "debunked", "untrue",
               "not true", "no such", "rumour", "rumor", "misinformation", "inaccurate", "no evidence",
               "has not", "have not", "did not", "does not", "not been", "no plan", "denied"]
MISLEADING_CUES = ["out of context", "missing context", "old photo", "old image", "old video", "reused",
                   "cropped", "re-captioned", "recaptioned", "misleading", "misrepresent", "does not show",
                   "unrelated event", "from 2019", "from 2020", "from 2021", "from 2022", "taken in 20",
                   "doctored", "edited image", "lacks context"]
ABSOLUTE_QUANTIFIERS = ["everyone", "every", "all", "entire", "whole", "nationwide", "100%", "always",
                        "each and every", "anyone", "everybody"]
LIMITING_QUALIFIERS = ["eligible", "only", "some", "certain", "selected", "select", "specific", "partial",
                       "pilot", "up to", "below", "small and marginal", "districts", "subject to",
                       "conditions", "registered", "qualifying", "five districts", "few"]
HEDGES = ["allegedly", "alleged", "reportedly", "reported to", "projected", "according to", "claimed",
          "claims", "purportedly", "unconfirmed", "sources said", "is said to", "appears to", "likely",
          "suspected", "believed to", "expected to", "estimated", "may", "might", "could"]
CERTAINTY_WORDS = ["confirmed", "proven", "proved", "definitely", "certainly", "undoubtedly",
                   "it is a fact", "established that", "without doubt"]
SENSATIONAL_CUES = ["shocker", "shocking", "you won't believe", "exposed", "secretly", "panic", "collapse",
                    "destroy", "danger", "urgent", "breaking!!", "must watch", "share before", "deleted",
                    "forward immediately", "viral", "bombshell", "miracle", "100% true", "they don't want",
                    "anonymous"]
ANTONYMS = [("cut", "unchanged"), ("raised", "unchanged"), ("hiked", "unchanged"), ("won", "lost"),
            ("approved", "rejected"), ("cancelled", "commission"), ("increase", "decrease"),
            ("rose", "fell"), ("open", "closed"), ("arrested", "released"), ("cut", "hiked")]

ABBREVIATIONS = ["Rs.", "Dr.", "Mr.", "Mrs.", "Ms.", "St.", "No.", "vs.", "Govt.", "Jan.", "Feb.", "Aug.",
                 "Sept.", "Sep.", "Oct.", "Nov.", "Dec.", "U.S.", "U.K.", "Prof.", "Gen.", "Lt.", "Col."]

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*%?")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]


def light_stem(tok: str) -> str:
    """Tiny, predictable suffix stripper (offices→office, taxes→tax, cancelled→cancel, collecting→collect)."""
    if tok[:1].isdigit() or len(tok) <= 4:
        return tok
    if tok.endswith("ies"):
        return tok[:-3] + "y"
    if tok.endswith("sses"):
        return tok[:-2]
    if tok.endswith("es") and tok[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return tok[:-2]
    for suf in ("ing", "ed"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            stem = tok[: -len(suf)]
            if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] not in "aeiouls":
                stem = stem[:-1]
            elif stem.endswith("ll") and len(stem) >= 5:
                stem = stem[:-1]
            return stem
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def content_tokens(text: str) -> list[str]:
    return [light_stem(t) for t in tokenize(text) if t not in STOPWORDS and t not in NEGATIONS and len(t) > 1]


_SENT_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”'’)]))\s+(?=[\"“'‘(₹]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    protected = text
    for i, abbr in enumerate(ABBREVIATIONS):
        protected = protected.replace(abbr, abbr.replace(".", f"<D{i}>"))
    protected = re.sub(r"\b([A-Z])\.(?=\s[A-Z])", r"\1<INI>", protected)  # initials: "V. Narayanan"
    parts = _SENT_SPLIT.split(protected)
    out = []
    for p in parts:
        for i, abbr in enumerate(ABBREVIATIONS):
            p = p.replace(abbr.replace(".", f"<D{i}>"), abbr)
        p = p.replace("<INI>", ".").strip()
        if p:
            out.append(p)
    return out


def word_count(text: str) -> int:
    return len((text or "").split())


# ------------------------------------------------------------------ quantities
MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
MONEY_RE = re.compile(
    r"(?:Rs\.?|INR|₹|\$|USD|NPR|£|€)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:crore|lakh|million|billion|trillion|cr|bn|mn))?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:crore|lakh|million|billion|trillion)(?:\s(?:rupees|dollars))?",
    re.I,
)
PCT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|per\s?cent|percent)", re.I)
DATE_RE = re.compile(
    rf"\b\d{{1,2}}\s{MONTHS}\.?(?:,?\s\d{{4}})?\b|\b{MONTHS}\s\d{{1,2}}(?:,\s?\d{{4}})?\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b(?:19|20)\d{{2}}\b"
)
NUM_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def extract_quantities(text: str) -> list[str]:
    """Money, percentages, dates and remaining bare numbers (verbatim strings)."""
    text = text or ""
    found: list[str] = []
    spans: list[tuple[int, int]] = []
    for rx in (MONEY_RE, PCT_RE, DATE_RE):
        for m in rx.finditer(text):
            if any(m.start() < e and m.end() > s for s, e in spans):
                continue
            spans.append((m.start(), m.end()))
            found.append(m.group(0).strip())
    for m in NUM_RE.finditer(text):
        if any(m.start() < e and m.end() > s for s, e in spans):
            continue
        spans.append((m.start(), m.end()))
        found.append(m.group(0))
    seen, out = set(), []
    for f in found:
        k = norm_quantity(f)
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def norm_quantity(q: str) -> str:
    q = q.lower().replace("per cent", "%").replace("percent", "%").replace("rs.", "rs").replace("₹", "rs")
    return re.sub(r"[\s,]", "", q)


def numbers_in(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in NUM_RE.finditer(text or "")}


def percentages_in(text: str) -> set[str]:
    return {re.sub(r"[^\d.]", "", m.group(0)) for m in PCT_RE.finditer(text or "")}


# ------------------------------------------------------------------ entities
ENTITY_STOP = {"The", "A", "An", "In", "On", "At", "It", "This", "That", "He", "She", "They", "We", "But",
               "And", "However", "According", "Officials", "Official", "Photo", "As", "If", "When", "While",
               "After", "Before", "Some", "Many", "Most", "Any", "Our", "Its", "There", "Here", "What", "Why",
               "How", "No", "Not", "Citizens", "Crowds", "Residents", "Police", "Government", "Ministry",
               "Show", "Tell", "Is", "Are", "Was", "Were", "Will", "Do", "Does", "Did", "Can", "Please", "Also",
               "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "SHOCKER",
               "BREAKING", "Fact", "Check", "Kindly", "Note", "I", "My", "You", "Your", "Ignore", "Latest", "Any",
               "Breaking", "Top", "Recent", "Today"}
GAZETTEER = {"Kerala", "ISRO", "RBI", "Reserve Bank", "Reserve Bank of India", "The Hindu", "Reuters",
             "Indian Express", "Times of India", "Hindustan Times", "BBC", "CNN", "Kantipur", "Koshi",
             "Chitradurga", "Sriharikota", "Election Commission", "Parliament", "Ministry of Finance",
             "Nepal", "India", "Kathmandu", "Delhi", "Mumbai", "Bihar", "Karnataka", "Pokhara", "IMD",
             "PM-KISAN", "Saptari", "Australia", "Zambia", "Western Ghats", "Lok Sabha", "MoF", "EC"}
_CAP_WORD = re.compile(r"^(?:[A-Z][a-zA-Z'’\-]*|[A-Z]{2,}[a-z]?s?|[A-Z]\.(?:[A-Z]\.)+)$")


def extract_entities(text: str) -> list[str]:
    """Heuristic capitalised-phrase NER with a small gazetteer."""
    text = text or ""
    ents: list[str] = []
    mid_caps = set()
    for sent in split_sentences(text):
        words = re.findall(r"[A-Za-z][A-Za-z'’\-.]*[A-Za-z]|[A-Z]", sent)
        for w in words[1:]:
            if _CAP_WORD.match(w):
                mid_caps.add(w)
    for sent in split_sentences(text):
        words = re.findall(r"[A-Za-z][A-Za-z'’\-.]*[A-Za-z.]|[A-Z]|\d+", sent)
        run: list[str] = []
        idx_start = 0

        def flush(run_, first_idx):
            if not run_:
                return
            while run_ and run_[-1] in ("of", "and", "the"):
                run_.pop()
            while run_ and run_[0] in ENTITY_STOP:
                run_.pop(0)
                first_idx += 1
            if not run_:
                return
            phrase = " ".join(w.rstrip(".") if not re.match(r"^[A-Z]\.", w) else w for w in run_)
            if len(run_) == 1 and first_idx == 0 and phrase not in mid_caps and phrase not in GAZETTEER \
                    and not phrase.isupper():
                return
            if len(phrase) > 1:
                ents.append(phrase)

        for i, w in enumerate(words):
            if _CAP_WORD.match(w) or (run and w in ("of", "and") and i + 1 < len(words) and _CAP_WORD.match(words[i + 1])):
                if not run:
                    idx_start = i
                run.append(w)
            else:
                flush(run, idx_start)
                run = []
        flush(run, idx_start)
    lowered = text.lower()
    for g in GAZETTEER:
        if re.search(r"\b" + re.escape(g.lower()) + r"\b", lowered) and g not in ents:
            if any(g in e for e in ents):
                continue
            ents.append(g)
    seen, out = set(), []
    for e in ents:
        if e.lower() not in seen:
            seen.add(e.lower())
            out.append(e)
    return out


def find_cues(text: str, cues: Iterable[str]) -> list[str]:
    low = " " + (text or "").lower() + " "
    hits = []
    for c in cues:
        pat = r"(?<![a-z])" + re.escape(c.lower()) + r"(?![a-z])"
        if re.search(pat, low):
            hits.append(c)
    return hits


def find_hedges(text: str) -> list[str]:
    return find_cues(text, HEDGES)


def has_negation(text: str) -> bool:
    toks = set(tokenize(text))
    return bool(toks & NEGATIONS) or "n't" in (text or "").lower()


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a and b else 0.0


def coverage(query_tokens: Iterable[str], doc_tokens: Iterable[str]) -> float:
    q, d = set(query_tokens), set(doc_tokens)
    return len(q & d) / len(q) if q else 0.0


def cosine_counts(a: list[str], b: list[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    num = sum(ca[k] * cb[k] for k in ca.keys() & cb.keys())
    den = math.sqrt(sum(v * v for v in ca.values())) * math.sqrt(sum(v * v for v in cb.values()))
    return num / den if den else 0.0


def content_hash(title: str, body: str) -> str:
    norm = re.sub(r"\W+", " ", f"{title} {body[:600]}".lower()).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def detect_language(text: str) -> str:
    if re.search(r"[ऀ-ॿ]", text or ""):
        return "ne"
    return "en"


def is_english(text: str, threshold: float = 0.85) -> bool:
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for c in letters if c.isascii())
    return ascii_letters / len(letters) >= threshold


# ------------------------------------------------------------------ taxonomy
TOPIC_LEXICON: dict[str, list[str]] = {
    "Tech": ["isro", "satellite", "launch", "rocket", "artificial intelligence", "software", "startup",
             "smartphone", "5g", "internet", "cyber", "chip", "technology", "tech", "app", "digital", "space",
             "orbit", "eos", "semiconductor", "ai"],
    "Economy": ["rbi", "repo", "inflation", "gdp", "budget", "tax", "allowance", "rupee", "market", "sensex",
                "bank", "finance", "economy", "economic", "trade", "investment", "scheme", "price", "stock",
                "monetary", "pm-kisan", "farmers", "solar park", "energy", "rs", "crore"],
    "Politics": ["parliament", "election", "voter", "minister", "party", "cabinet", "policy", "bill", "mla",
                 "lok sabha", "vote", "campaign", "poll", "mayoral", "election commission", "session",
                 "opposition", "constituency", "voter roll"],
    "Society": ["flood", "floods", "monsoon", "rain", "rainfall", "landslide", "health", "hospital", "school",
                "education", "police", "crime", "community", "disaster", "relief", "water", "heatwave",
                "red alert", "evacuat", "danger level", "river", "students"],
    "World": ["united nations", "global", "international", "summit", "climate", "war", "europe", "foreign",
              "bilateral", "world", "zambia", "cop", "treaty", "g20"],
    "Sports": ["cricket", "football", "match", "series", "tournament", "olympic", "world cup", "goal",
               "wicket", "player", "team", "coach", "ipl", "t20", "innings", "medal"],
    "Opinion": ["opinion", "column", "editorial", "we must", "i believe", "perspective", "view:"],
    "Entertainment": ["film", "movie", "bollywood", "actor", "actress", "music", "box office", "ott",
                      "celebrity", "award", "cinema", "festival", "album", "director"],
}


def topic_scores(title: str, body: str = "") -> dict[str, float]:
    t, b = " " + (title or "").lower() + " ", " " + (body or "").lower() + " "
    scores = {}
    for topic, kws in TOPIC_LEXICON.items():
        s = 0.0
        for kw in kws:
            pat = r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])" if kw != "evacuat" else "evacuat"
            s += 3 * len(re.findall(pat, t)) + len(re.findall(pat, b))
        scores[topic] = s
    return scores


def classify_topic(title: str, body: str = "", default: str = "Society") -> str:
    scores = topic_scores(title, body)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else default


# Concept lexicon for the offline "semantic" fallback: paraphrases share a concept token.
CONCEPTS: dict[str, list[str]] = {
    "monsoon": ["monsoon", "rain", "rainfall", "downpour", "flood", "floods", "flooding", "deluge", "inundation",
                "waterlogging", "cloudburst", "red alert", "landslide", "danger level", "submerged"],
    "space": ["isro", "rocket", "satellite", "launch", "orbit", "spacecraft", "sriharikota", "eos", "space"],
    "monetary": ["rbi", "repo", "interest rate", "rate cut", "central bank", "inflation", "monetary", "reserve bank"],
    "election": ["vote", "voter", "voters", "poll", "ballot", "electoral", "election", "voter roll", "by-election"],
    "cricket": ["cricket", "t20", "odi", "wicket", "innings", "series", "batsman", "bowler", "runs"],
    "film": ["film", "movie", "cinema", "box office", "actor", "actress", "director", "screening"],
    "welfare": ["allowance", "scheme", "subsidy", "cash transfer", "relief", "pm-kisan", "handout", "benefit"],
    "legislature": ["parliament", "session", "lok sabha", "assembly", "bill", "house", "speaker"],
    "energy": ["solar", "power", "energy", "electricity", "mw", "renewable", "grid"],
    "climate": ["climate", "emissions", "cop", "warming", "carbon", "summit"],
    "health": ["health", "hospital", "vaccine", "doctor", "disease", "virus", "medicine", "patients"],
    "misinfo": ["fake", "hoax", "viral", "rumour", "rumor", "misinformation", "fact-check", "debunk", "false claim"],
    "economy": ["gdp", "growth", "economy", "markets", "sensex", "stocks", "investment", "trade", "exports"],
    "tech": ["ai", "artificial intelligence", "software", "startup", "chip", "semiconductor", "smartphone", "5g"],
}


def concept_expand(text: str) -> str:
    low = " " + (text or "").lower() + " "
    extra = []
    for concept, terms in CONCEPTS.items():
        hits = sum(1 for term in terms if re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", low))
        if hits:
            extra.extend([f"concept_{concept}"] * min(hits, 3))
    return (text or "") + " " + " ".join(extra)
