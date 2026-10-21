"""Dataset management for the three IEEE Dataport sources.

1. NewsSumm (TrendyBits News Summarization Dataset)     columns: Source, Content, Summary
2. LIAR (PolitiFact, 6-class claim level)              14 columns, labels pants-fire … true
3. Integrated ISOT + BharatFakeNewsKosh (binary, article level) columns: date, title, subject, text, label

Real files (CSV/TSV/XLSX/JSON) are imported with `import_*` functions. When no local file is
present, `generate_*` builds a SYNTHETIC benchmark with the exact same schema so the repository
runs immediately after cloning. Synthetic rows are template-generated with label noise and
are NOT the IEEE Dataport data — metrics measured on them are pipeline sanity benchmarks only.

Every dataset goes through `audit()` (dedup, missing fields, English filter, label normalisation)
and `stratified_split()` (80/10/10) BEFORE any model fitting or prompt tuning. The test partition
is fingerprinted (SHA-256 of its row ids) in a manifest so later stages can prove it was not used
for tuning.
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .text_utils import is_english

LIAR_LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
LIAR_COLUMNS = ["id", "label", "statement", "subject", "speaker", "job_title", "state", "party",
                "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts",
                "pants_on_fire_counts", "context"]
NEWSSUMM_COLUMNS = ["Source", "Content", "Summary"]
INTEGRATED_COLUMNS = ["date", "title", "subject", "text", "label"]

SCHEMAS = {
    "newssumm": {"columns": NEWSSUMM_COLUMNS, "label": None, "required": NEWSSUMM_COLUMNS,
                 "url": "https://ieee-dataport.org/documents/trendybits-news-summarization-dataset"},
    "liar": {"columns": LIAR_COLUMNS, "label": "label", "required": ["id", "label", "statement"],
             "url": "https://ieee-dataport.org/documents/trendybits-fake-news-dataset"},
    "integrated": {"columns": INTEGRATED_COLUMNS, "label": "label", "required": ["title", "text", "label"],
                   "url": "https://ieee-dataport.org/documents/trendybits-fake-news-dataset"},
}

LIAR_LABEL_MAP = {
    "pants-fire": "pants-fire", "pants on fire": "pants-fire", "pants-on-fire": "pants-fire", "pants fire": "pants-fire",
    "false": "false", "barely-true": "barely-true", "barely true": "barely-true", "half-true": "half-true",
    "half true": "half-true", "mostly-true": "mostly-true", "mostly true": "mostly-true", "true": "true",
}
BINARY_LABEL_MAP = {"1": 1, "true": 1, "real": 1, "verified": 1, "0": 0, "false": 0, "fake": 0, "misleading": 0}


def dataset_dir(name: str) -> Path:
    d = config.DATASETS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================== loaders
def read_any(path: str | Path, header: bool = True) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if suf in (".json", ".jsonl"):
        return pd.read_json(p, lines=suf == ".jsonl")
    sep = "\t" if suf in (".tsv", ".txt") else ","
    return pd.read_csv(p, sep=sep, header=0 if header else None, quoting=3 if sep == "\t" else 0,
                       on_bad_lines="skip", encoding_errors="replace")


def _rename(df: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    ren = {}
    for target, cands in mapping.items():
        for c in [target.lower(), *cands]:
            if c in lower:
                ren[lower[c]] = target
                break
    return df.rename(columns=ren)


def import_newssumm(paths: list[str | Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = _rename(read_any(p), {"Source": ["source", "publisher", "dataset", "origin"],
                                   "Content": ["content", "article", "text", "document", "body"],
                                   "Summary": ["summary", "highlights", "reference", "abstract"]})
        if "Source" not in df.columns:
            df["Source"] = Path(p).stem
        frames.append(df[[c for c in NEWSSUMM_COLUMNS if c in df.columns]])
    return pd.concat(frames, ignore_index=True)


def import_liar(paths: list[str | Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = read_any(p, header=False)
        if df.shape[1] >= 14 and str(df.iloc[0, 1]).lower().strip() in LIAR_LABEL_MAP:
            df = df.iloc[:, :14]
            df.columns = LIAR_COLUMNS
        else:  # file with a header row
            df = _rename(read_any(p), {"statement": ["claim", "text"], "job_title": ["job", "speaker_job"],
                                       "state": ["state_info"], "party": ["party_affiliation"]})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def import_integrated(isot_true: str | None = None, isot_fake: str | None = None,
                      bharat: str | None = None, combined: str | None = None) -> pd.DataFrame:
    mapping = {"title": ["headline", "statement", "news_title"], "text": ["content", "body", "article", "news_body", "eng_trans_news_body"],
               "subject": ["category", "topic", "news_category"], "date": ["publish_date", "published", "news_date"],
               "label": ["class", "target", "is_true", "label_bin"]}
    frames = []
    if combined:
        frames.append(_rename(read_any(combined), mapping))
    if isot_true:
        df = _rename(read_any(isot_true), mapping)
        df["label"] = 1
        frames.append(df)
    if isot_fake:
        df = _rename(read_any(isot_fake), mapping)
        df["label"] = 0
        frames.append(df)
    if bharat:
        df = _rename(read_any(bharat), mapping)
        if "title" not in df.columns and "text" in df.columns:
            df["title"] = df["text"].astype(str).str.slice(0, 90)
        if "subject" not in df.columns:
            df["subject"] = "BharatFakeNewsKosh"
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for c in INTEGRATED_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    return out[INTEGRATED_COLUMNS]


# =============================================================================== audit + split
def audit(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    schema = SCHEMAS[name]
    rep = {"dataset": name, "rows_in": int(len(df))}
    for c in schema["columns"]:
        if c not in df.columns:
            df[c] = "" if c not in ("barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts",
                                    "pants_on_fire_counts") else 0
    df = df[schema["columns"]].copy()
    # missing required fields
    req = schema["required"]
    miss = df[req].isna().any(axis=1) | (df[req].astype(str).apply(lambda s: s.str.strip() == "")).any(axis=1)
    rep["removed_missing"] = int(miss.sum())
    df = df[~miss]
    # label normalisation / consistency
    if name == "liar":
        lab = df["label"].astype(str).str.lower().str.strip().map(LIAR_LABEL_MAP)
        rep["removed_bad_label"] = int(lab.isna().sum())
        df = df.assign(label=lab)[lab.notna()]
        for c in ["barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        for c in ["subject", "speaker", "job_title", "state", "party", "context"]:
            df[c] = df[c].fillna("unknown").astype(str)
        text_col = "statement"
    elif name == "integrated":
        lab = df["label"].astype(str).str.lower().str.strip().map(BINARY_LABEL_MAP)
        rep["removed_bad_label"] = int(lab.isna().sum())
        df = df.assign(label=lab)[lab.notna()]
        df["label"] = df["label"].astype(int)
        text_col = "text"
    else:
        rep["removed_bad_label"] = 0
        text_col = "Content"
    # English-language filter
    eng = df[text_col].astype(str).apply(is_english)
    rep["removed_non_english"] = int((~eng).sum())
    df = df[eng]
    # duplicate removal (exact text duplicates, also catches train/test leakage)
    key = df[text_col].astype(str).str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    dup = key.duplicated()
    rep["removed_duplicates"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    if name != "liar":
        df.insert(0, "row_id", [f"{name}-{i:06d}" for i in range(len(df))])
    else:
        df["id"] = df["id"].astype(str)
    rep["rows_out"] = int(len(df))
    if schema["label"]:
        rep["label_distribution"] = {str(k): int(v) for k, v in df[schema["label"]].value_counts().sort_index().items()}
    return df, rep


def stratified_split(df: pd.DataFrame, label_col: str | None, seed: int = config.SEED,
                     ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict[str, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    strat = df[label_col] if label_col else None
    train, rest = train_test_split(df, test_size=ratios[1] + ratios[2], random_state=seed, stratify=strat)
    strat_rest = rest[label_col] if label_col else None
    valid, test = train_test_split(rest, test_size=ratios[2] / (ratios[1] + ratios[2]), random_state=seed,
                                   stratify=strat_rest)
    return {"train": train.reset_index(drop=True), "valid": valid.reset_index(drop=True), "test": test.reset_index(drop=True)}


def save_splits(name: str, splits: dict[str, pd.DataFrame], audit_report: dict, origin: str) -> dict:
    d = dataset_dir(name)
    id_col = "id" if name == "liar" else "row_id"
    for k, part in splits.items():
        part.to_csv(d / f"{k}.csv", index=False)
    test_ids = ",".join(sorted(splits["test"][id_col].astype(str)))
    manifest = {
        "dataset": name, "origin": origin, "schema": SCHEMAS[name]["columns"], "ieee_dataport": SCHEMAS[name]["url"],
        "split_ratios": [0.8, 0.1, 0.1], "seed": config.SEED, "stratified_on": SCHEMAS[name]["label"],
        "sizes": {k: int(len(v)) for k, v in splits.items()},
        "test_sha256": hashlib.sha256(test_ids.encode()).hexdigest(), "test_isolated": True,
        "audit": audit_report,
    }
    if SCHEMAS[name]["label"]:
        manifest["label_distribution"] = {k: {str(a): int(b) for a, b in v[SCHEMAS[name]["label"]].value_counts().sort_index().items()}
                                          for k, v in splits.items()}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_split(name: str, split: str) -> pd.DataFrame:
    p = dataset_dir(name) / f"{split}.csv"
    if not p.exists():
        ensure_datasets()
    df = pd.read_csv(p, keep_default_na=False)
    return df


def load_manifest(name: str) -> dict:
    p = dataset_dir(name) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def ensure_datasets(force: bool = False, sizes: dict | None = None) -> dict[str, dict]:
    """Make sure audited, split versions of all three datasets exist (generate if absent)."""
    sizes = sizes or {"newssumm": 700, "liar": 4000, "integrated": 4000}
    manifests = {}
    for name, gen in (("newssumm", generate_newssumm), ("liar", generate_liar), ("integrated", generate_integrated)):
        d = dataset_dir(name)
        if force or not (d / "test.csv").exists():
            raw = gen(sizes[name])
            clean, rep = audit(raw, name)
            manifests[name] = save_splits(name, stratified_split(clean, SCHEMAS[name]["label"]), rep,
                                          origin="synthetic-benchmark (schema-faithful, template generated)")
        else:
            manifests[name] = load_manifest(name)
    bench = config.DATASETS_DIR / "retrieval_benchmark.json"
    if force or not bench.exists():
        build_retrieval_benchmark()
    return manifests


# =============================================================================== synthetic generators
PUBLISHERS = ["The Hindu", "Indian Express", "Hindustan Times", "Times of India", "CNN", "Daily Mail", "Reuters",
              "Multi-News", "XSum"]
PLACES = ["Kerala", "Assam", "Bihar", "Odisha", "Uttarakhand", "Karnataka", "Maharashtra", "Gujarat", "Koshi Province",
          "Tamil Nadu"]
PEOPLE = ["Anita Rao", "Rahul Menon", "Priya Sharma", "Vikram Singh", "Meera Iyer", "Arjun Das", "Sunita Thapa",
          "Kiran Patel", "Deepak Joshi", "Farah Khan", "Rohan Gupta", "Lakshmi Nair"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def _date_str(rng: random.Random) -> tuple[str, str]:
    d = date(2026, 10, 20) - timedelta(days=rng.randint(0, 120))
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}", d.isoformat()


def _fill(rng: random.Random) -> dict:
    ds, iso = _date_str(rng)
    return dict(
        place=rng.choice(PLACES), person=rng.choice(PEOPLE), person2=rng.choice(PEOPLE), date=ds, iso=iso,
        n=rng.randint(3, 14), mm=rng.randint(120, 340), evac=f"{rng.randint(2, 60) * 100:,}", camps=rng.randint(12, 180),
        crore=rng.choice([25, 50, 75, 100, 150, 250, 400]), pct=f"{rng.choice([4, 5, 5.5, 6, 6.25, 6.5, 7])}",
        infl=f"{rng.choice([3.1, 3.6, 4.2, 4.8, 5.4, 5.9])}", bps=rng.choice([25, 50]),
        sat=f"{rng.choice(['EOS', 'GSAT', 'RISAT', 'Cartosat'])}-{rng.randint(2, 30):02d}",
        rocket=f"{rng.choice(['PSLV', 'GSLV', 'LVM3'])}-{rng.choice(['C', 'F', 'M'])}{rng.randint(40, 70)}",
        mins=rng.randint(16, 22), runs=rng.randint(45, 140), wkts=rng.randint(3, 8), margin=rng.randint(2, 9),
        team=rng.choice(["India", "Nepal", "Australia", "England", "Sri Lanka", "South Africa"]),
        opp=rng.choice(["Pakistan", "New Zealand", "Bangladesh", "West Indies", "Afghanistan", "Ireland"]),
        film=rng.choice(["Monsoon Diaries", "The Last Train", "Kaagaz Ke Phool 2", "Neon Nights", "River Song", "Echoes"]),
        gross=rng.choice([42, 68, 95, 120, 155, 210]), countries=rng.randint(80, 196), bn=rng.choice([100, 250, 300, 500]),
        year=rng.choice([2030, 2035, 2040]), company=rng.choice(["Krutrim Labs", "AgriSense", "PayNest", "Medify", "VoltEdge"]),
        mn=rng.choice([12, 25, 40, 60, 110]), kids=f"{rng.randint(10, 90) * 1000:,}", seats=rng.randint(240, 400),
        votes=rng.randint(280, 420),
    )


FILLERS = {
    "Society": ["Several roads were blocked by landslides and power supply was disrupted in some areas.",
                "Schools remained closed as a precaution.", "Forecasters expect conditions to ease by the weekend.",
                "Local volunteers joined rescue teams in the worst-hit villages."],
    "Economy": ["Markets reacted cautiously to the announcement.", "Economists had widely expected the decision.",
                "Bond yields were little changed after the statement.", "Analysts said the outlook remained uncertain."],
    "Tech": ["The launch was broadcast live on the agency's website.", "Engineers had completed final checks overnight.",
             "Industry experts welcomed the development.", "The company plans to expand its team next year."],
    "Politics": ["Opposition leaders criticised the timing of the move.", "The debate lasted for more than six hours.",
                 "Political analysts said the decision could shape the coming elections."],
    "Sports": ["The stadium was packed with more than 40,000 fans.", "Rain delayed the start of play by an hour.",
               "The coach praised the team's fielding effort."],
    "Entertainment": ["Critics gave the film mixed reviews.", "The soundtrack has trended on streaming platforms.",
                      "The cast thanked fans at a press event."],
    "World": ["Negotiations ran past the scheduled deadline.", "Activists staged protests outside the venue.",
              "Observers called the outcome a modest step forward."],
}

FAMILIES = [
    dict(family="floods", topic="Society",
         title="Floods hit {n} districts of {place} after heavy monsoon rain",
         lead="Heavy monsoon rain lashed {place} on {date}, triggering floods in {n} districts.",
         key=["The India Meteorological Department recorded {mm} mm of rainfall in 24 hours.",
              "Officials said {evac} people had been moved to {camps} relief camps."],
         quote="\"Our teams are working round the clock,\" said {person}, the state disaster management chief.",
         outcome="The state government announced Rs {crore} crore for immediate relief.",
         summary="Floods affected {n} districts in {place} following heavy monsoon rain on {date}. Around {evac} people were shifted to {camps} relief camps and Rs {crore} crore was announced for relief."),
    dict(family="monetary", topic="Economy",
         title="Central bank keeps repo rate at {pct}% as inflation eases",
         lead="The Reserve Bank of India kept the repo rate unchanged at {pct} per cent on {date}.",
         key=["Retail inflation eased to {infl} per cent last month, data showed.",
              "Governor {person} said growth remained resilient."],
         quote="\"Price stability remains our priority,\" {person} told reporters.",
         outcome="The central bank retained its neutral stance and said it would watch food prices closely.",
         summary="The RBI held the repo rate at {pct} per cent on {date} as inflation eased to {infl} per cent. It kept a neutral stance."),
    dict(family="launch", topic="Tech",
         title="ISRO launches {sat} satellite aboard {rocket}",
         lead="ISRO launched the {sat} satellite from Sriharikota on {date} aboard the {rocket} rocket.",
         key=["The satellite was placed in orbit about {mins} minutes after lift-off.",
              "Mission director {person} said all systems were performing normally."],
         quote="\"This mission strengthens our earth observation capability,\" said {person}.",
         outcome="The satellite will support agriculture, forestry and disaster monitoring.",
         summary="ISRO's {rocket} placed the {sat} satellite in orbit {mins} minutes after its {date} launch from Sriharikota. It will support agriculture and disaster monitoring."),
    dict(family="cricket", topic="Sports",
         title="{team} beat {opp} by {margin} wickets",
         lead="{team} beat {opp} by {margin} wickets on {date} to take the series lead.",
         key=["{person} top-scored with {runs} runs off 52 balls.",
              "The bowlers took {wkts} wickets in the first innings."],
         quote="\"It was a complete team performance,\" said captain {person2}.",
         outcome="The final match of the series will be played next week.",
         summary="{team} defeated {opp} by {margin} wickets on {date}, with {person} scoring {runs} runs. The series finale is next week."),
    dict(family="film", topic="Entertainment",
         title="'{film}' earns Rs {gross} crore in opening weekend",
         lead="The film '{film}' earned Rs {gross} crore at the box office in its opening weekend, trade analysts said on {date}.",
         key=["Directed by {person}, the film released on more than 3,000 screens.",
              "Lead actor {person2} said the response had been overwhelming."],
         quote="\"Audiences have embraced the story,\" said {person}.",
         outcome="Distributors expect the film to cross Rs 300 crore worldwide.",
         summary="'{film}', directed by {person}, collected Rs {gross} crore in its opening weekend. Distributors expect it to cross Rs 300 crore."),
    dict(family="climate", topic="World",
         title="{countries} countries agree ${bn} billion climate finance goal",
         lead="Negotiators from {countries} countries agreed on {date} to mobilise ${bn} billion a year in climate finance by {year}.",
         key=["Developing nations said the amount fell short of their needs.",
              "Summit president {person} called the deal a turning point."],
         quote="\"We have kept the 1.5 degree goal alive,\" {person} said.",
         outcome="The agreement will be reviewed at next year's summit.",
         summary="{countries} countries agreed to mobilise ${bn} billion a year in climate finance by {year}. Developing nations said it was not enough."),
    dict(family="startup", topic="Tech",
         title="{company} raises ${mn} million in funding round",
         lead="{company} said on {date} it had raised ${mn} million in a funding round.",
         key=["Chief executive {person} said the money will fund expansion into five new markets.",
              "The company currently serves more than two million users."],
         quote="\"We are just getting started,\" said {person}.",
         outcome="The startup plans to double its engineering team by next year.",
         summary="{company} raised ${mn} million on {date} to expand into five new markets, CEO {person} said."),
    dict(family="vaccination", topic="Society",
         title="{place} vaccinates {kids} children in measles drive",
         lead="Health workers in {place} vaccinated {kids} children against measles in a week-long drive that ended on {date}.",
         key=["District health officer {person} said coverage reached {n}0 per cent in rural areas.",
              "Mobile teams visited remote villages to reach missed children."],
         quote="\"No child should be left behind,\" said {person}.",
         outcome="A second phase is planned for next month.",
         summary="A measles drive in {place} reached {kids} children by {date}. A second phase is planned next month."),
    dict(family="parliament", topic="Politics",
         title="Lok Sabha passes data protection amendment bill",
         lead="The Lok Sabha passed the data protection amendment bill on {date} with {votes} members voting in favour.",
         key=["Minister {person} said the bill strengthens user consent rules.",
              "Opposition members walked out before the vote."],
         quote="\"Citizens deserve control over their data,\" said {person}.",
         outcome="The bill now goes to the Rajya Sabha.",
         summary="The Lok Sabha passed the data protection amendment bill on {date} with {votes} votes; it now moves to the Rajya Sabha."),
]


def _make_article(rng: random.Random) -> dict:
    fam = rng.choice(FAMILIES)
    f = _fill(rng)
    key = [k.format(**f) for k in fam["key"]]
    fillers = rng.sample(FILLERS.get(fam["topic"], FILLERS["World"]), k=min(len(FILLERS.get(fam["topic"], FILLERS["World"])), rng.randint(2, 3)))
    body = [fam["lead"].format(**f)]
    middle = fillers[:1] + key[:1] + [fam["quote"].format(**f)] + fillers[1:] + key[1:]
    rng.shuffle(middle)
    body += middle + [fam["outcome"].format(**f)]
    src = rng.choice(PUBLISHERS)
    return dict(Source=src, Content=" ".join(body), Summary=fam["summary"].format(**f),
                _title=fam["title"].format(**f), _topic=fam["topic"], _family=fam["family"],
                _place=f["place"], _person=f["person"], _iso=f["iso"])


def generate_newssumm(n: int = 700, seed: int = config.SEED, with_meta: bool = False) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = [_make_article(rng) for _ in range(n)]
    df = pd.DataFrame(rows)
    # inject a few audit targets: duplicates, missing summary, non-English row
    dirty = df.sample(6, random_state=seed)
    df = pd.concat([df, dirty], ignore_index=True)
    df.loc[len(df)] = dict(Source="Hindi Wire", Content="केरल में भारी बारिश से बाढ़ आई।", Summary="बाढ़", _title="", _topic="Society",
                           _family="floods", _place="Kerala", _person="", _iso="2026-10-01")
    df.loc[len(df)] = dict(Source="Reuters", Content="Incomplete row without summary.", Summary="", _title="",
                           _topic="World", _family="", _place="", _person="", _iso="2026-10-01")
    return df if with_meta else df[NEWSSUMM_COLUMNS]


def build_retrieval_benchmark(n_docs: int = 160, seed: int = config.SEED + 7) -> dict:
    """Synthetic retrieval benchmark with graded relevance judgements (held out from tuning)."""
    rng = random.Random(seed)
    docs = []
    for i in range(n_docs):
        a = _make_article(rng)
        docs.append(dict(id=f"rb-{i:04d}", title=a["_title"], body=a["Content"], source=a["Source"], domain="",
                         topic=a["_topic"], family=a["_family"], place=a["_place"], person=a["_person"],
                         published_at=a["_iso"], kind="news", url=f"https://bench.trendybits.local/{i}"))
    para = {
        "floods": "heavy downpour and inundation in {place}", "vaccination": "measles immunisation campaign {place}",
        "monetary": "interest rate decision by the central bank", "launch": "space agency rocket mission",
        "cricket": "{team} cricket win", "film": "box office collections of new movie",
        "climate": "climate finance deal at summit", "startup": "startup funding news", "parliament": "lower house passes bill",
    }
    queries = []
    seen = set()
    for d in docs:
        fam = d["family"]
        key = (fam, d["place"] if fam in ("floods", "vaccination") else "")
        if key in seen:
            continue
        seen.add(key)
        if fam in ("floods", "vaccination"):
            rel = [x["id"] for x in docs if x["family"] == fam and x["place"] == d["place"]]
            q = para[fam].format(place=d["place"])
        elif fam == "cricket":
            continue
        else:
            rel = [x["id"] for x in docs if x["family"] == fam]
            q = para[fam]
        if len(rel) >= 2:
            queries.append(dict(qid=f"q{len(queries):03d}", type="semantic", query=q, relevant=rel, profile=None))
    profiles = [["Tech", "Economy"], ["Sports", "Entertainment"], ["Society", "Politics"], ["World", "Tech"],
                ["Economy", "Politics"], ["Society", "Sports"]]
    for j, topics in enumerate(profiles):
        rel = [x["id"] for x in docs if x["topic"] in topics]
        queries.append(dict(qid=f"p{j:03d}", type="personalized", query="latest top stories for me", relevant=rel,
                            profile=dict(user_id=f"bench{j}", topics=topics, sources=[], entities=[])))
    bench = {"docs": docs, "queries": queries, "note": "synthetic; relevance = same event family (+ place) or profile topic"}
    (config.DATASETS_DIR / "retrieval_benchmark.json").write_text(json.dumps(bench, indent=1))
    return bench


def load_retrieval_benchmark() -> dict:
    p = config.DATASETS_DIR / "retrieval_benchmark.json"
    if not p.exists():
        return build_retrieval_benchmark()
    return json.loads(p.read_text())


# ------------------------------------------------------------------------- LIAR (6-class)
LIAR_SUBJECTS = {
    "economy": ["the unemployment rate fell to {pct} percent", "the state economy grew {pct} percent last year",
                "wages have risen {pct} percent since {year}", "manufacturing jobs increased by {num} thousand"],
    "taxes": ["{opp} voted to raise taxes {n} times", "our state cut property taxes {n} times since {year}",
              "the new tax plan raises taxes on the middle class", "corporate tax revenue dropped {pct} percent"],
    "health-care": ["the health care law cut Medicare by {num} billion dollars", "insurance premiums doubled since {year}",
                    "{num} thousand residents lost their coverage", "the vaccine program saved {num} million dollars"],
    "immigration": ["border crossings rose {pct} percent this year", "the state spends {num} million dollars on undocumented immigrants",
                    "{opp} supports open borders", "deportations fell to a ten-year low"],
    "education": ["graduation rates rose to {pct} percent", "teacher salaries are the lowest in the nation",
                  "the state cut school funding by {num} million dollars", "class sizes doubled since {year}"],
    "crime": ["the crime rate in {city} has doubled", "violent crime fell {pct} percent since {year}",
              "the city hired {num} new police officers", "gun deaths rose {pct} percent last year"],
    "energy": ["gas prices rose {pct} percent under {opp}", "the state produces {pct} percent of its power from wind",
               "the pipeline will create {num} thousand jobs", "solar capacity tripled since {year}"],
    "elections": ["{num} thousand dead people voted in the last election", "voter turnout reached {pct} percent",
                  "{opp} raised {num} million dollars from lobbyists", "mail ballots were counted twice"],
    "federal-budget": ["the deficit grew {pct} percent under {opp}", "foreign aid is {pct} percent of the budget",
                       "the stimulus cost {num} billion dollars", "the budget is balanced for the first time since {year}"],
}
LOW_CUES = ["every single", "never", "100 percent", "secretly", "completely destroyed", "the worst in history",
            "nobody", "all of them", "hoax", "totally", "always", "millions of", "anonymous sources say", "banned forever",
            "shocking", "they are hiding"]
PANTS_CUES = ["microchips", "secret plot", "aliens", "mind control", "forward immediately", "cover-up", "fake crisis"]
MID_CUES = ["nearly", "about", "more than", "some", "roughly", "in some areas", "almost", "partly"]
HIGH_CUES = ["according to the Bureau of Labor Statistics", "data from the Census Bureau show",
             "per the Congressional Budget Office", "according to official figures", "state records show",
             "the latest annual report shows", "a nonpartisan audit found"]
CONTEXTS_BY_TRUTH = {
    "low": ["a Facebook post", "a chain email", "a viral image", "a tweet", "a campaign rally", "a radio ad"],
    "mid": ["a TV interview", "a debate", "a campaign ad", "a speech", "a news conference"],
    "high": ["a press release", "a floor speech", "an op-ed", "a committee hearing", "a news conference"],
}
PARTIES = ["democrat", "republican", "independent", "none"]
STATES = ["Texas", "Florida", "Ohio", "New York", "California", "Georgia", "Wisconsin", "Virginia", "Arizona", ""]
JOBS = ["U.S. Senator", "Governor", "State Representative", "Congressman", "Mayor", "Talk show host", "Activist", ""]
CITIES = ["Chicago", "Austin", "Miami", "Detroit", "Atlanta", "Phoenix"]


def generate_liar(n: int = 4000, seed: int = config.SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    speakers = []
    for i in range(60):
        t = float(nrng.beta(2, 2))
        party = rng.choice(PARTIES[:3])
        speakers.append(dict(name=f"speaker-{i:02d}", t=t, party=party, state=rng.choice(STATES), job=rng.choice(JOBS)))
    for name in ["viral-image", "facebook-posts", "chain-email", "bloggers"]:
        speakers.append(dict(name=name, t=float(nrng.uniform(0.02, 0.2)), party="none", state="", job=""))
    rows = []
    for i in range(n):
        sp = rng.choice(speakers)
        idx = int(np.clip(round(sp["t"] * 5 + nrng.normal(0, 1.35)), 0, 5))
        label = LIAR_LABELS[idx]
        subject = rng.choice(list(LIAR_SUBJECTS))
        core = rng.choice(LIAR_SUBJECTS[subject]).format(
            pct=rng.choice([3, 4.2, 5, 7.5, 9, 12, 15, 18, 22, 25, 33, 40, 60]), year=rng.randint(2004, 2015),
            num=rng.randint(5, 900), n=rng.randint(2, 30),
            opp=rng.choice(["my opponent", "the governor", "the president", "Congress", "the mayor", "the senator"]),
            city=rng.choice(CITIES))
        core += rng.choice(["", "", f" in {rng.choice(STATES[:-1])}", " this year", " over the last decade",
                            " under the current administration", f" since {rng.randint(2005, 2015)}"])
        # label-correlated cues with neighbour-label noise
        cue_idx = idx if rng.random() > 0.3 else int(np.clip(idx + rng.choice([-1, 1]), 0, 5))
        parts = [core]
        if rng.random() < 0.65:
            if cue_idx <= 1:
                parts.insert(0, rng.choice(LOW_CUES).capitalize() + ",") if rng.random() < 0.5 else parts.append(rng.choice(LOW_CUES))
                if cue_idx == 0 and rng.random() < 0.6:
                    parts.append("— " + rng.choice(PANTS_CUES))
            elif cue_idx <= 3:
                parts.insert(0, rng.choice(MID_CUES).capitalize())
            else:
                parts.insert(0, rng.choice(HIGH_CUES).capitalize() + ",")
        statement = " ".join(parts).strip()
        statement = statement[0].upper() + statement[1:] + "."
        bucket = "low" if idx <= 1 else ("mid" if idx <= 3 else "high")
        ctx = rng.choice(CONTEXTS_BY_TRUTH[bucket] if rng.random() < 0.6 else sum(CONTEXTS_BY_TRUTH.values(), []))
        total = 0 if sp["party"] == "none" and rng.random() < 0.5 else rng.randint(1, 60)
        probs = np.exp(-((np.arange(6) - sp["t"] * 5) ** 2) / 4.0)
        counts = nrng.multinomial(total, probs / probs.sum()) if total else np.zeros(6, dtype=int)
        rows.append(dict(id=f"{i + 1}.json", label=label, statement=statement, subject=subject, speaker=sp["name"],
                         job_title=sp["job"], state=sp["state"], party=sp["party"],
                         barely_true_counts=int(counts[2]), false_counts=int(counts[1]), half_true_counts=int(counts[3]),
                         mostly_true_counts=int(counts[4]), pants_on_fire_counts=int(counts[0]), context=ctx))
    df = pd.DataFrame(rows)
    # audit targets: label variants, a duplicate and a missing statement
    df.loc[0, "label"] = "Pants on Fire"
    df.loc[1, "label"] = "TRUE" if df.loc[1, "label"] == "true" else df.loc[1, "label"]
    df = pd.concat([df, df.iloc[[5, 6]]], ignore_index=True)
    df.loc[len(df)] = dict(id="99999.json", label="half-true", statement="", subject="economy", speaker="x", job_title="",
                           state="", party="none", barely_true_counts=0, false_counts=0, half_true_counts=0,
                           mostly_true_counts=0, pants_on_fire_counts=0, context="")
    return df


# ------------------------------------------------------------------------- Integrated binary
TRUE_SUBJECTS_ISOT = ["politicsNews", "worldnews"]
FAKE_SUBJECTS_ISOT = ["News", "politics", "left-news", "Government News", "US_News"]
BHARAT_SUBJECTS = ["Politics", "Health", "Society", "Viral", "Economy"]
CITIES_DATELINE = ["WASHINGTON", "NEW DELHI", "LONDON", "KATHMANDU", "MUMBAI", "BEIJING", "BRUSSELS"]
TRUE_LEADS = [
    "{org} said on {day} it would {action}, according to a statement.",
    "Officials at {org} confirmed on {day} that {fact}.",
    "{org} reported that {fact}, citing official data released on {day}.",
    "Lawmakers approved a measure on {day} under which {fact}, the {org} spokesperson said.",
]
TRUE_BODY = [
    "The figures were in line with estimates from economists polled earlier this month.",
    "A spokesperson declined to comment further on the details.",
    "The ministry said the plan would be reviewed after six months.",
    "Data published by the statistics office showed a {pct} percent change from a year earlier.",
    "The decision follows weeks of negotiations between the two sides.",
    "Opposition lawmakers said they would scrutinise the proposal.",
    "The report was released after a review by independent auditors.",
]
FAKE_LEADS = [
    "BREAKING: {person} EXPOSED as {org} secretly plans to {bad_action}!",
    "SHOCKER: {org} is hiding the truth — {bad_fact}!",
    "You won't believe what {person} just did — {bad_fact}.",
    "WATCH: Leaked video proves {bad_fact}, mainstream media silent!",
]
FAKE_BODY = [
    "Share this before it is deleted!", "Forward immediately to all your WhatsApp groups.",
    "An anonymous source told us everything.", "The mainstream media will never tell you this.",
    "Doctors are shocked by this simple trick.", "They don't want you to know the truth.",
    "This is 100% true and confirmed by insiders.", "Wake up people, the cover-up is real!",
]
ORGS = ["the finance ministry", "the health ministry", "the election commission", "the central bank", "the city council",
        "the World Health Organization", "the education department", "the state government"]
ACTIONS = ["raise fuel subsidies", "extend the vaccination programme", "publish revised voter lists",
           "review interest rates next month", "expand rural road funding", "hire 5,000 new teachers"]
FACTS = ["inflation eased to 4.1 percent in September", "exports rose 6 percent year on year",
         "voter registration closes on 30 November", "the budget deficit narrowed to 5.6 percent of GDP",
         "rainfall was 12 percent above average", "hospital admissions fell for a third week"]
BAD_ACTIONS = ["ban cash nationwide tomorrow", "put microchips in vaccines", "cancel all elections",
               "seize everyone's bank savings", "give Rs 50,000 to every citizen", "shut down the internet forever"]
SHARED_CLAIMS = ["fuel prices will change from next month", "a new welfare scheme was announced for farmers",
                 "the exam schedule was revised", "the dam's gates were opened after heavy rain",
                 "the minister resigned after the meeting", "a curfew was imposed in two districts",
                 "the bank merged two of its branches", "train fares will be revised from Monday"]
SHARED_ACTIONS = ["revise train fares", "announce a farmers' scheme", "change the exam schedule",
                  "impose a curfew in two districts"]
BAD_FACTS = ["the new notes contain GPS tracking chips", "drinking hot water cures every virus",
             "salt prices will rise 500 percent tonight", "the moon landing video was shot in a studio",
             "every citizen will get free gold coins", "the bridge collapse was planned by foreign agents"]


def generate_integrated(n: int = 4000, seed: int = config.SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        is_true = rng.random() < 0.5
        bharat = rng.random() < 0.35
        d = date(2016, 1, 1) + timedelta(days=rng.randint(0, 700)) if not bharat else date(2019, 1, 1) + timedelta(days=rng.randint(0, 1600))
        fill = dict(org=rng.choice(ORGS), action=rng.choice(ACTIONS), fact=rng.choice(FACTS), day=rng.choice(["Monday", "Tuesday", "Thursday", "Friday"]),
                    person=rng.choice(PEOPLE), bad_action=rng.choice(BAD_ACTIONS), bad_fact=rng.choice(BAD_FACTS),
                    pct=rng.choice([2, 3.5, 6, 8.2, 11]))
        hard = rng.random() < 0.15  # stylistic noise: hard examples written in the "other" register
        style_true = is_true != hard
        if rng.random() < 0.3:  # claim whose truth is NOT inferable from the wording (shared pool)
            shared = rng.choice(SHARED_CLAIMS)
            fill.update(fact=shared, bad_fact=shared, action=rng.choice(SHARED_ACTIONS))
            fill["bad_action"] = fill["action"]
        if style_true and is_true and rng.random() < 0.3:  # ordinary wire-style news report (ISOT "true" register)
            art = _make_article(rng)
            title, body = art["_title"], [art["Content"]]
        elif style_true:
            lead = rng.choice(TRUE_LEADS).format(**fill)
            if not bharat and rng.random() < 0.6:
                lead = f"{rng.choice(CITIES_DATELINE)} (Reuters) - " + lead
            body = [lead] + [s.format(**fill) for s in rng.sample(TRUE_BODY, 3)]
            title = lead.split(",")[0][:90].replace(" (Reuters) - ", ": ")
            if not is_true:  # false claim in sober language: swap in a fabricated fact
                body[0] = body[0].replace(fill["fact"], fill["bad_fact"]).replace(fill["action"], fill["bad_action"])
        else:
            lead = rng.choice(FAKE_LEADS).format(**fill)
            body = [lead] + rng.sample(FAKE_BODY, 3)
            title = lead[:90]
            if is_true:  # true story with click-bait framing
                body.insert(1, f"Officials confirmed that {fill['fact']}.")
        subject = rng.choice(BHARAT_SUBJECTS) if bharat else rng.choice(TRUE_SUBJECTS_ISOT if is_true else FAKE_SUBJECTS_ISOT)
        label = int(is_true) if rng.random() > 0.03 else int(not is_true)  # ~3% annotation noise
        rows.append(dict(date=d.isoformat(), title=title, subject=subject, text=" ".join(body), label=label))
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype(object)
    df.loc[3, "label"] = "True"
    df.loc[4, "label"] = "fake"
    df = pd.concat([df, df.iloc[[10, 11, 12]]], ignore_index=True)
    df.loc[len(df)] = dict(date="2020-02-02", title="वायरल दावा", subject="Viral", text="यह खबर पूरी तरह झूठी है और व्हाट्सएप पर फैलाई जा रही है।", label=0)
    return df
