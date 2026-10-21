import React from "react";

/** LIME token heatmap: renders the analysed text with signed attribution pills. */
export function LimeHeatmap({ lime, maxChars = 900 }) {
  if (!lime || !lime.tokens) return null;
  // reports saved by older versions carry no `text`: rebuild it from the token offsets
  const text =
    lime.text ||
    lime.tokens.reduce((acc, t) => acc.padEnd(t.start, " ").slice(0, t.start) + t.text, "");
  const pieces = [];
  let cursor = 0;
  const tokens = [...lime.tokens].sort((a, b) => a.start - b.start);
  tokens.forEach((t, i) => {
    if (t.start >= maxChars) return;
    if (t.start > cursor) pieces.push(<span key={`g${i}`}>{text.slice(cursor, t.start)}</span>);
    const n = t.normalized || 0;
    if (Math.abs(n) < 0.05) {
      pieces.push(<span key={`t${i}`}>{t.text}</span>);
    } else {
      const alpha = 0.18 + 0.62 * Math.min(1, Math.abs(n));
      const bg = n < 0 ? `rgba(220, 38, 38, ${alpha})` : `rgba(22, 163, 74, ${alpha})`;
      pieces.push(
        <span key={`t${i}`} className="lime-pill" style={{ background: bg }} title={`LIME weight ${t.weight > 0 ? "+" : ""}${t.weight}`}>
          {t.text}
          <sup>{t.weight > 0 ? "+" : ""}{t.weight.toFixed(2)}</sup>
        </span>
      );
    }
    cursor = t.end;
  });
  if (cursor < Math.min(text.length, maxChars)) pieces.push(<span key="tail">{text.slice(cursor, maxChars)}</span>);
  if (text.length > maxChars) pieces.push(<span key="more">…</span>);
  return (
    <div>
      <div className="lime-text">{pieces}</div>
      <div className="lime-lists">
        <div>
          <b className="neg">Pushes toward False / Misleading</b>
          {lime.top_negative?.length ? (
            lime.top_negative.map((t) => (
              <span key={t.token} className="chip chip-neg">
                {t.token} {t.weight.toFixed(3)}
              </span>
            ))
          ) : (
            <span className="muted"> none</span>
          )}
        </div>
        <div>
          <b className="pos">Pushes toward Verified</b>
          {lime.top_positive?.length ? (
            lime.top_positive.map((t) => (
              <span key={t.token} className="chip chip-pos">
                {t.token} +{t.weight.toFixed(3)}
              </span>
            ))
          ) : (
            <span className="muted"> none</span>
          )}
        </div>
      </div>
      <div className="muted small">
        {lime.num_samples} perturbations · {lime.num_features} features · local fidelity R² = {lime.local_fidelity_r2} · black box: {lime.black_box}
      </div>
    </div>
  );
}

/** SHAP bars: signed contribution of each pipeline dimension to the final credibility score. */
export function ShapBars({ shap, compact = false }) {
  if (!shap) return null;
  const maxAbs = Math.max(1, ...shap.contributions.map((c) => Math.abs(c.contribution)));
  return (
    <div className="shap">
      <div className="shap-head muted small">
        Baseline prior {shap.baseline_score} {shap.contributions.map((c) => (c.contribution >= 0 ? " + " : " − ") + Math.abs(c.contribution)).join("")} ={" "}
        <b>{shap.final_score}</b>
      </div>
      {shap.contributions.map((c) => {
        const w = (Math.abs(c.contribution) / maxAbs) * 50;
        return (
          <div className="shap-row" key={c.key}>
            <div className="shap-label">
              {c.label}
              {!compact && <span className="muted small"> (value {c.value}, prior {c.baseline_value})</span>}
            </div>
            <div className="shap-track">
              <div className="shap-zero" />
              <div
                className={`shap-bar ${c.contribution >= 0 ? "pos" : "neg"}`}
                style={c.contribution >= 0 ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }}
              />
            </div>
            <div className={`shap-val ${c.contribution >= 0 ? "pos" : "neg"}`}>
              {c.contribution >= 0 ? "+" : ""}
              {c.contribution.toFixed(2)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function XaiPanel({ xai, title = "🧠 Explainable AI (XAI) attribution" }) {
  if (!xai) return null;
  return (
    <section className="card xai-card">
      <h3>{title}</h3>
      <div className="xai-grid">
        <div>
          <h4>LIME token heatmap</h4>
          <LimeHeatmap lime={xai.lime} />
        </div>
        <div>
          <h4>SHAP dimension contributions</h4>
          <ShapBars shap={xai.shap} />
          <div className="muted small">{xai.aggregator}</div>
          <div className="muted small">Efficiency check: baseline + Σφ − final = {xai.shap?.efficiency_gap}</div>
        </div>
      </div>
    </section>
  );
}
