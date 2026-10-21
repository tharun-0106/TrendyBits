import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import ReportCard from "../components/ReportCard.jsx";
import { DemoTag, ErrorBox, Spinner, StatusPill, VERDICT_CLASS, fmt } from "../components/ui.jsx";

function splitSentences(text) {
  return (text || "").match(/[^.!?]+[.!?]+["”']?\s*|[^.!?]+$/g) || [];
}

export default function ArticlePage({ id, go, metrics }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [quick, setQuick] = useState("");
  const [rated, setRated] = useState(false);

  const load = () =>
    api
      .article(id)
      .then(setData)
      .catch((e) => setError(e.message));
  useEffect(() => {
    setData(null);
    setError(null);
    setRated(false);
    load();
    window.scrollTo(0, 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  async function runAnalyzer() {
    setBusy(true);
    try {
      await api.analyze({ input_type: "article", article_id: id });
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner text="Loading article…" />;
  const { article, summary, report, related, source } = data;
  const claimBySentence = {};
  (report?.claim_results || []).forEach((c) => {
    claimBySentence[c.text.replace(/^Photo: /, "").trim()] = c;
  });
  const summ = metrics?.models?.find((m) => m.key === "summarizer")?.metrics || {};
  const crumb = article.extra?.breadcrumb || article.title.slice(0, 48);
  const pull = report?.claim_results?.find((c) => c.type === "factual");

  return (
    <div className="article-layout">
      <div className="article-main">
        <div className="breadcrumb">
          <a onClick={() => go("home", {})}>Home</a> / <a onClick={() => go("home", { topic: article.topic })}>{article.topic}</a> / {crumb}
        </div>
        <div className="pills">
          <span className="pill pill-cat">{(article.topic || "News").toUpperCase()}</span>
          {report ? (
            <a href="#report">
              <StatusPill status={report.status} score={report.overall_score} />
              <span className="muted small"> — report below</span>
            </a>
          ) : (
            <StatusPill />
          )}
          <DemoTag demo={article.demo} />
        </div>
        <h1 className="headline">{article.title}</h1>
        <div className="byline">
          By {article.author || "Staff"} · {article.source} · {article.published_at} ·{" "}
          <a href={article.url} target="_blank" rel="noreferrer">
            original article ↗
          </a>
        </div>

        <section className="card summary-card">
          <div className="summary-head">
            <span className="eyebrow">Pipeline A · {summary.model_used} · faithful summary</span>
            <span className="badges">
              {summ.rougeL !== undefined && <span className="chip">ROUGE-L {fmt(summ.rougeL)}</span>}
              {summ.faithfulness !== undefined && <span className="chip">Faithfulness {fmt(summ.faithfulness)}</span>}
              <span className="chip">entities kept {fmt(summary.guard.entity_preservation, 2)}</span>
              <span className="chip chip-pos">no verdict ✓</span>
              {summary.chunked && <span className="chip">{summary.chunks_processed} chunks</span>}
            </span>
          </div>
          <p>{summary.summary}</p>
          <ul>
            {summary.key_points.map((k, i) => (
              <li key={i}>{k}</li>
            ))}
          </ul>
          <div className="muted small">
            {summary.attribution} ·{" "}
            <a href={summary.source.url} target="_blank" rel="noreferrer">
              {summary.source.headline}
            </a>{" "}
            · The summary reports what the article says; it makes no true/false judgement.
          </div>
          {!rated && (
            <div className="rate">
              Was this summary faithful?{" "}
              {[1, 2, 3, 4, 5].map((n) => (
                <button key={n} className="chip chip-btn" onClick={() => api.feedback({ target_type: "summary", target_id: id, rating: n }).then(() => setRated(true))}>
                  {n}★
                </button>
              ))}
            </div>
          )}
        </section>

        {pull && <blockquote className={`pullquote ${VERDICT_CLASS[pull.verdict]}`}>“{pull.text}”</blockquote>}
        <div className="article-body">
          {splitSentences(article.body).map((s, i) => {
            const c = claimBySentence[s.trim()];
            return c ? (
              <mark key={i} className={`claim-mark ${VERDICT_CLASS[c.verdict]}`} title={`${c.verdict}: ${c.rationale}`}>
                {s}
              </mark>
            ) : (
              <span key={i}>{s}</span>
            );
          })}
          {article.image_caption && <figure className="caption">📷 {article.image_caption}</figure>}
        </div>

        {report ? (
          <ReportCard report={report} related={related} onOpenArticle={(rid) => go("article", { id: rid })} />
        ) : (
          <section className="card">
            <p>This article has not been analysed yet.</p>
            <button className="btn btn-primary" onClick={runAnalyzer} disabled={busy}>
              {busy ? "Analysing…" : "⚡ Run independent analyzer"}
            </button>
          </section>
        )}
      </div>

      <aside className="sidebar">
        <div className="card sticky">
          <h4>IN THIS ARTICLE</h4>
          {report ? (
            <ol className="toc">
              {report.claim_results.map((c, i) => (
                <li key={c.cid}>
                  <a href={`#claim-${c.cid}`}>
                    Claim {i + 1} <span className={`badge ${VERDICT_CLASS[c.verdict]}`}>{c.verdict}</span>
                  </a>
                </li>
              ))}
            </ol>
          ) : (
            <p className="muted small">No report yet.</p>
          )}
          <h4>SOURCE BOX</h4>
          {source || report?.source_info ? (
            <dl className="source-box">
              <dt>Outlet</dt>
              <dd>{(source || report.source_info).name || report?.source_info?.publisher}</dd>
              <dt>Trust</dt>
              <dd>{report ? `${report.source_trust.score}/100 (${report.source_trust.label})` : `${source.trust_score}/100`}</dd>
              <dt>Owner</dt>
              <dd>{(source || report.source_info).owner}</dd>
              <dt>Masthead</dt>
              <dd>{(source || report.source_info).masthead_verified ? "verified ✓" : "not found ✗"}</dd>
              <dt>Corrections</dt>
              <dd>{(source || report.source_info).corrections_record}</dd>
              <dt>History</dt>
              <dd>{(source || report.source_info).history}</dd>
              <dt>Contact</dt>
              <dd>{(source || report.source_info).contact}</dd>
            </dl>
          ) : (
            <p className="muted small">Outlet not in registry.</p>
          )}
          <h4>CHECK ANOTHER CLAIM</h4>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (quick.trim()) go("factcheck", { tab: "text", content: quick, autorun: true });
            }}
          >
            <textarea className="input" rows={3} value={quick} onChange={(e) => setQuick(e.target.value)} placeholder="Paste a claim…" />
            <button className="btn btn-primary wide" type="submit">
              ⚡ Check
            </button>
          </form>
        </div>
      </aside>
    </div>
  );
}
