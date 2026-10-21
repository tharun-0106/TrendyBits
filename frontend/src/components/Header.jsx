import React, { useState } from "react";

export const CATEGORIES = ["Politics", "Society", "Economy", "World", "Sports", "Opinion", "Entertainment", "Tech"];

export default function Header({ theme, onToggleTheme, lang, setLang, models, ticker, page, go, onSearch, onScrape, scraping }) {
  const [q, setQ] = useState("");
  const today = new Date().toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric" });
  const ollama = models?.ollama?.reachable;
  const nav = [
    { key: "home", label: "News", params: {} },
    ...CATEGORIES.map((c) => ({ key: "home", label: c, params: { topic: c } })),
    { key: "factcheck", label: "⚡ Fact-Check", params: {}, cls: "nav-accent" },
    { key: "chat", label: "💬 AI Chatbot", params: {} },
    { key: "admin", label: "📊 Admin: Model Metrics & XAI", params: {} },
    { key: "profile", label: "👤 Profile", params: {} },
  ];
  const active = (n) => n.key === page.name && (n.key !== "home" || (n.params.topic || null) === (page.params?.topic || null));

  return (
    <header>
      <div className="utility">
        <div className="wrap utility-row">
          <span>{today}</span>
          <span className="sep">|</span>
          <a href="#" onClick={(e) => e.preventDefault()}>E-Paper</a>
          <span className="sep">|</span>
          <span className="lang">
            <button className={lang === "en" ? "on" : ""} onClick={() => setLang("en")}>EN</button> /{" "}
            <button className={lang === "ne" ? "on" : ""} onClick={() => setLang("ne")}>NE</button>
          </span>
          <span className="sep">|</span>
          <button className="theme-toggle" onClick={onToggleTheme} title="Toggle dark / light mode">
            {theme === "dark" ? "☀️ Light" : "🌙 Dark"}
          </button>
          <span className="grow" />
          <span className="models-pill" title={ollama ? "Ollama reachable" : "Ollama offline — deterministic fallback engine active"}>
            Active Models: {models?.summarizer || "qwen:4b"} + {models?.reasoner || "deepseek-r1:8b"}
            <span className={`dot ${ollama ? "dot-on" : "dot-off"}`} /> {ollama ? "Ollama live" : "offline fallback"}
          </span>
        </div>
      </div>
      <div className="wrap brand-row">
        <button className="brand" onClick={() => go("home", {})}>
          <span className="logo">TB</span>
          <span>
            <span className="brand-name">TrendyBits</span>
            <span className="brand-tag">Personalised news · Independent fact-checks</span>
          </span>
        </button>
        <form
          className="search"
          onSubmit={(e) => {
            e.preventDefault();
            onSearch(q);
          }}
        >
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search news… (semantic)" aria-label="Search news" />
          <button type="submit">🔍</button>
        </form>
        <button className="btn btn-ghost" onClick={onScrape} disabled={scraping}>
          {scraping ? "Scraping…" : "↻ Live RSS Scrape"}
        </button>
        <button className="btn btn-primary" onClick={() => go("factcheck", {})}>
          ✓ Check News
        </button>
      </div>
      <nav className="catnav">
        <div className="wrap catnav-row">
          {nav.map((n) => (
            <button key={n.label} className={`${n.cls || ""} ${active(n) ? "active" : ""}`} onClick={() => go(n.key, n.params)}>
              {n.label}
            </button>
          ))}
        </div>
      </nav>
      <div className="ticker">
        <div className="wrap ticker-row">
          <span className="ticker-label">BREAKING</span>
          <div className="ticker-viewport">
            <div className="ticker-track">
              {[...(ticker || []), ...(ticker || [])].map((t, i) => (
                <span key={i}>
                  {t} <span className="ticker-sep">—</span>
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
