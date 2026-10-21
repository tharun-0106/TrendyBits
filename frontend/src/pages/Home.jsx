import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { DemoTag, ErrorBox, Spinner, StatusPill } from "../components/ui.jsx";

export default function Home({ topic, query, go }) {
  const [articles, setArticles] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setArticles(null);
    setError(null);
    api
      .articles(topic, query)
      .then((r) => setArticles(r.articles))
      .catch(setError);
  }, [topic, query]);

  if (error) return <ErrorBox error={`${error.message} — is the backend running on :8000?`} />;
  if (!articles) return <Spinner text="Loading stories…" />;
  const [lead, ...rest] = articles;
  return (
    <div className="home">
      <div className="section-title">
        {query ? (
          <>
            Search results for “{query}” <span className="muted small">Sentence-BERT/FAISS + MMR, personalised for your profile</span>
          </>
        ) : (
          topic || "Top stories"
        )}
      </div>
      {!articles.length && <p className="muted">No stories yet in this section. Try the ↻ Live RSS Scrape button.</p>}
      {lead && (
        <article className="lead-story" onClick={() => go("article", { id: lead.id })}>
          <div className="pills">
            <span className="pill pill-cat">{(lead.topic || "News").toUpperCase()}</span>
            {lead.status !== undefined && <StatusPill status={lead.status} score={lead.score} />}
            <DemoTag demo={lead.demo} />
          </div>
          <h1>{lead.title}</h1>
          <p>{lead.snippet}</p>
          <div className="byline">
            {lead.source} · {lead.published_at}
          </div>
        </article>
      )}
      <div className="grid">
        {rest.map((a) => (
          <article className="story" key={a.id} onClick={() => go("article", { id: a.id })}>
            <div className="pills">
              <span className="pill pill-cat">{(a.topic || "News").toUpperCase()}</span>
              {a.status !== undefined && <StatusPill status={a.status} score={a.score} />}
              <DemoTag demo={a.demo} />
            </div>
            <h3>{a.title}</h3>
            <p className="muted">{a.snippet?.slice(0, 140)}…</p>
            <div className="byline">
              {a.source} · {a.published_at}
              {a.reasons?.length ? <span className="muted small"> · {a.reasons.join(", ")}</span> : null}
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
