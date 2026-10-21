"""Module 2a — Dynamic news scraping (RSS feeds + arbitrary user URLs).

* RSS via feedparser (falls back to a BeautifulSoup XML parser when feedparser is not installed).
* Full-text extraction via newspaper3k when installed, else requests + BeautifulSoup heuristics
  (og: meta tags, <article> paragraphs, JSON-LD dates).
* Standardised fields: title, body, source, domain, published_at, url, topic, author.
* HTML cleaning, English/Nepali language tag, de-duplication (URL + normalised content hash),
  SSRF guard, and indirect-prompt-injection scanning before anything is stored.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import requests

from . import database as db
from .security import is_safe_url, scan
from .text_utils import classify_topic, detect_language

log = logging.getLogger("trendybits.scraper")
UA = {"User-Agent": "Mozilla/5.0 (TrendyBits research crawler; +https://github.com/trendybits)"}

RSS_FEEDS = {
    "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
    "Indian Express": "https://indianexpress.com/feed/",
    "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "BBC": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Reuters": "https://www.reutersagency.com/feed/?best-topics=top-news&post_type=best",
}


def clean_html(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    for t in soup(["script", "style", "noscript", "iframe", "form", "nav", "footer", "aside"]):
        t.decompose()
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def _iso(dt_str: str | None) -> str | None:
    if not dt_str:
        return None
    for fn in (lambda s: parsedate_to_datetime(s), lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))):
        try:
            return fn(dt_str).date().isoformat()
        except (TypeError, ValueError, IndexError):
            continue
    m = re.search(r"\d{4}-\d{2}-\d{2}", dt_str)
    return m.group(0) if m else None


def parse_feed(xml: str) -> list[dict]:
    try:
        import feedparser  # type: ignore

        feed = feedparser.parse(xml)
        return [dict(title=e.get("title", ""), url=e.get("link"), summary=clean_html(e.get("summary", "")),
                     published=e.get("published") or e.get("updated"), author=e.get("author")) for e in feed.entries]
    except ImportError:  # stdlib RSS 2.0 / Atom fallback
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
        except ET.ParseError:
            return []

        def local(tag: str) -> str:
            return tag.rsplit("}", 1)[-1].lower()

        items = []
        for node in root.iter():
            if local(node.tag) not in ("item", "entry"):
                continue
            f = {local(c.tag): c for c in node}
            link = f.get("link")
            href = (link.get("href") or (link.text or "").strip()) if link is not None else None
            desc = f.get("description") or f.get("summary") or f.get("content")
            pub = f.get("pubdate") or f.get("published") or f.get("updated")
            items.append(dict(title=(f["title"].text or "").strip() if "title" in f else "", url=href,
                              summary=clean_html(desc.text or "") if desc is not None else "",
                              published=(pub.text or "").strip() if pub is not None else None, author=None))
        return items


def fetch_article(url: str, timeout: float = 12.0) -> dict:
    """Download and extract a single article page."""
    if not is_safe_url(url):
        raise ValueError("URL rejected by SSRF guard (only public http/https hosts are allowed)")
    try:
        from newspaper import Article  # type: ignore

        art = Article(url)
        art.download()
        art.parse()
        if art.text and len(art.text) > 200:
            return dict(title=art.title, body=art.text, author=", ".join(art.authors) or None,
                        published_at=art.publish_date.date().isoformat() if art.publish_date else None,
                        image_caption=None)
    except Exception:  # newspaper3k missing or failed -> heuristic extractor
        pass
    from bs4 import BeautifulSoup

    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    def meta(*names):
        for n in names:
            t = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n})
            if t and t.get("content"):
                return t["content"].strip()
        return None

    title = meta("og:title", "twitter:title") or (soup.title.get_text(strip=True) if soup.title else url)
    container = soup.find("article") or soup.find("main") or soup
    paras = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    body = " ".join(p for p in paras if len(p.split()) > 6)
    if len(body) < 200:
        body = clean_html(str(container))[:8000]
    return dict(title=title, body=body[:20000], author=meta("author", "article:author"),
                published_at=_iso(meta("article:published_time", "og:updated_time", "date", "pubdate")),
                image_caption=meta("og:image:alt"))


def standardise(raw: dict, url: str, source: str | None = None) -> dict:
    host = (urlparse(url).hostname or "").removeprefix("www.")
    reg = db.get_source(host)
    flags = scan(raw.get("body", ""))
    return dict(title=raw.get("title", "").strip()[:300], body=raw.get("body", "").strip(), url=url,
                source=source or (reg["name"] if reg else host), domain=host,
                published_at=raw.get("published_at") or datetime.now(timezone.utc).date().isoformat(),
                topic=classify_topic(raw.get("title", ""), raw.get("body", "")), author=raw.get("author"),
                kind="news" if (reg and reg["trust_score"] >= 60) else "web", image_caption=raw.get("image_caption"),
                language=detect_language(raw.get("body", "")),
                extra={"scraped_at": datetime.now(timezone.utc).isoformat(), "injection_flags": [f["kind"] for f in flags]})


def scrape_url(url: str) -> dict:
    try:
        raw = fetch_article(url)
    except (requests.RequestException, ValueError) as exc:
        return {"article_id": None, "error": str(exc)}
    if len(raw.get("body", "")) < 80:
        return {"article_id": None, "error": "No article text could be extracted from the page"}
    art = standardise(raw, url)
    aid, inserted = db.upsert_article(art)
    return {"article_id": aid, "inserted": inserted, "duplicate": not inserted, "title": art["title"],
            "injection_flags": art["extra"]["injection_flags"]}


def scrape_rss(max_per_feed: int = 8, fetch_full: bool = False, feeds: dict | None = None) -> dict:
    stats = {"feeds": 0, "fetched": 0, "inserted": 0, "duplicates": 0, "errors": [], "items": []}
    for name, feed_url in (feeds or RSS_FEEDS).items():
        try:
            r = requests.get(feed_url, headers=UA, timeout=10)
            r.raise_for_status()
            entries = parse_feed(r.text)[:max_per_feed]
            stats["feeds"] += 1
        except requests.RequestException as exc:
            stats["errors"].append(f"{name}: {exc.__class__.__name__}")
            continue
        for e in entries:
            if not e.get("url") or not e.get("title"):
                continue
            stats["fetched"] += 1
            body = e.get("summary") or ""
            if fetch_full:
                try:
                    body = fetch_article(e["url"]).get("body") or body
                except (requests.RequestException, ValueError):
                    pass
            if len(body) < 40:
                body = f"{e['title']}. {body}"
            art = standardise(dict(title=e["title"], body=body, published_at=_iso(e.get("published")), author=e.get("author")),
                              e["url"], source=name)
            aid, inserted = db.upsert_article(art)
            stats["inserted" if inserted else "duplicates"] += 1
            if inserted:
                stats["items"].append({"id": aid, "title": art["title"], "source": name})
    return stats
