import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { CATEGORIES } from "../components/Header.jsx";
import { ErrorBox, Spinner } from "../components/ui.jsx";

const SOURCES = ["The Hindu", "Indian Express", "Hindustan Times", "Times of India", "Reuters", "BBC", "Kantipur", "The Kathmandu Post"];

export default function Profile({ userId, setUserId }) {
  const [p, setP] = useState(null);
  const [error, setError] = useState(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setP(null);
    api.profile(userId).then(setP).catch((e) => setError(e.message));
  }, [userId]);

  if (error) return <ErrorBox error={error} />;
  if (!p) return <Spinner text="Loading profile…" />;
  const toggle = (key, v) => setP({ ...p, [key]: p[key].includes(v) ? p[key].filter((x) => x !== v) : [...p[key], v] });

  async function save() {
    const r = await api.saveProfile(userId, { name: p.name, topics: p.topics, sources: p.sources, entities: p.entities, language: p.language });
    setP({ ...p, ...r });
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
  }

  return (
    <div className="profile">
      <section className="card">
        <h2>👤 Reader profile</h2>
        <label className="field">
          Active user
          <select value={userId} onChange={(e) => setUserId(e.target.value)}>
            <option value="demo">demo (Tech + Economy reader)</option>
            <option value="guest">guest (sparse profile)</option>
          </select>
        </label>
        <h4>Preferred topics</h4>
        <div className="checks">
          {CATEGORIES.map((c) => (
            <label key={c}>
              <input type="checkbox" checked={p.topics.includes(c)} onChange={() => toggle("topics", c)} /> {c}
            </label>
          ))}
        </div>
        <h4>Preferred sources</h4>
        <div className="checks">
          {SOURCES.map((s) => (
            <label key={s}>
              <input type="checkbox" checked={p.sources.includes(s)} onChange={() => toggle("sources", s)} /> {s}
            </label>
          ))}
        </div>
        <label className="field">
          Followed entities (comma separated)
          <input className="input" value={p.entities.join(", ")} onChange={(e) => setP({ ...p, entities: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) })} />
        </label>
        <button className="btn btn-primary" onClick={save}>
          {saved ? "Saved ✓" : "Save preferences"}
        </button>
        <p className="muted small">Preferences boost retrieval scores (+0.12 topic, +0.08 source, +0.05 entity); MMR keeps the feed diverse.</p>
      </section>
      <section className="card">
        <h3>Recent conversational queries</h3>
        <ul>
          {(p.recent_queries || []).map((q) => (
            <li key={q.id}>
              <b>{q.query}</b> <span className="muted small">→ {q.intent.intent} · {q.created_at}</span>
            </li>
          ))}
          {!p.recent_queries?.length && <li className="muted">No queries yet — try the chatbot.</li>}
        </ul>
      </section>
    </div>
  );
}
