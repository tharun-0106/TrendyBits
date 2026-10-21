import React, { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import Header from "./components/Header.jsx";
import Home from "./pages/Home.jsx";
import FactCheck from "./pages/FactCheck.jsx";
import ArticlePage from "./pages/ArticlePage.jsx";
import Admin from "./pages/Admin.jsx";
import Chatbot from "./pages/Chatbot.jsx";
import Profile from "./pages/Profile.jsx";

function readStore(key, fallback) {
  try {
    return window.localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
}
function writeStore(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* storage unavailable (private mode) */
  }
}

class PageBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidUpdate(prev) {
    if (prev.pageKey !== this.props.pageKey && this.state.error) this.setState({ error: null });
  }
  render() {
    if (this.state.error) {
      return <div className="error-box">⚠ This page failed to render: {String(this.state.error.message || this.state.error)}</div>;
    }
    return this.props.children;
  }
}

function parseHash() {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  switch (parts[0]) {
    case "article":
      return { name: "article", params: { id: parts[1] } };
    case "topic":
      return { name: "home", params: { topic: parts[1] } };
    case "search":
      return { name: "home", params: { q: parts[1] } };
    case "factcheck":
    case "chat":
    case "admin":
    case "profile":
      return { name: parts[0], params: {} };
    default:
      return { name: "home", params: {} };
  }
}

function toHash(name, params = {}) {
  if (name === "article") return `#/article/${encodeURIComponent(params.id)}`;
  if (name === "home" && params.topic) return `#/topic/${encodeURIComponent(params.topic)}`;
  if (name === "home" && params.q) return `#/search/${encodeURIComponent(params.q)}`;
  if (name === "home") return "#/";
  return `#/${name}`;
}

export default function App() {
  const [theme, setTheme] = useState(() => readStore("tb-theme", window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  const [lang, setLang] = useState("en");
  const [page, setPage] = useState(parseHash);
  const [models, setModels] = useState(null);
  const [ticker, setTicker] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [userId, setUserId] = useState(() => readStore("tb-user", "demo"));
  const [toast, setToast] = useState(null);
  const [scraping, setScraping] = useState(false);
  const transient = useRef({});

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    writeStore("tb-theme", theme);
  }, [theme]);
  useEffect(() => {
    writeStore("tb-user", userId);
  }, [userId]);

  useEffect(() => {
    api.models().then(setModels).catch(() => {});
    api.ticker().then((t) => setTicker(t.items)).catch(() => {});
    api.adminMetrics().then(setMetrics).catch(() => {});
    const onHash = () => {
      const p = parseHash();
      setPage({ name: p.name, params: { ...p.params, ...transient.current } });
      transient.current = {};
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = useCallback((name, params = {}) => {
    const hash = toHash(name, params);
    transient.current = params;
    if (window.location.hash === hash) {
      setPage({ name, params });
      transient.current = {};
    } else {
      window.location.hash = hash;
    }
    window.scrollTo(0, 0);
  }, []);

  const flash = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(null), 4500);
  };

  async function scrape() {
    setScraping(true);
    try {
      const s = await api.scrapeRss();
      flash(`RSS scrape: ${s.feeds} feeds, ${s.fetched} items, ${s.inserted} new, ${s.duplicates} duplicates` + (s.errors.length ? ` · ${s.errors.length} feed errors (offline?)` : ""));
      if (s.inserted) go("home", {});
    } catch (e) {
      flash(`Scrape failed: ${e.message}`);
    } finally {
      setScraping(false);
    }
  }

  let body;
  switch (page.name) {
    case "article":
      body = <ArticlePage id={page.params.id} go={go} metrics={metrics} />;
      break;
    case "factcheck":
      body = <FactCheck key={JSON.stringify(page.params)} models={models} go={go} initial={page.params} />;
      break;
    case "chat":
      body = <Chatbot go={go} userId={userId} />;
      break;
    case "admin":
      body = <Admin onMetrics={setMetrics} />;
      break;
    case "profile":
      body = <Profile userId={userId} setUserId={setUserId} />;
      break;
    default:
      body = <Home topic={page.params.topic} query={page.params.q} go={go} />;
  }

  return (
    <div className="app" lang={lang}>
      <Header
        theme={theme}
        onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
        lang={lang}
        setLang={setLang}
        models={models}
        ticker={ticker}
        page={page}
        go={go}
        onSearch={(q) => (q.trim() ? go("home", { q: q.trim() }) : go("home", {}))}
        onScrape={scrape}
        scraping={scraping}
      />
      {lang === "ne" && <div className="wrap lang-note">नेपाली इन्टरफेस (बेटा): समाचार सामग्री मूल भाषामै देखाइन्छ; विश्लेषकले NE / EN स्वतः पहिचान गर्छ।</div>}
      <main className="wrap">
        <PageBoundary pageKey={`${page.name}:${JSON.stringify(page.params)}`}>{body}</PageBoundary>
      </main>
      {toast && <div className="toast">{toast}</div>}
      <footer className="footer">
        <div className="wrap">
          TrendyBits · Pipeline A (retrieval + faithful summaries) never judges truth · Pipeline B (independent credibility analysis) cites only
          retrieved evidence · Scores are system estimates, not final truth labels.
        </div>
      </footer>
    </div>
  );
}
