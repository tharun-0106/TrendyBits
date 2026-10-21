import React from "react";

export const VERDICT_CLASS = {
  FALSE: "v-false",
  MISLEADING: "v-misleading",
  UNVERIFIED: "v-unverified",
  SUPPORTED: "v-supported",
};

export function VerdictBadge({ verdict }) {
  return <span className={`badge ${VERDICT_CLASS[verdict] || "v-unverified"}`}>{verdict}</span>;
}

export function scoreClass(score) {
  if (score === null || score === undefined) return "s-none";
  if (score >= 65) return "s-good";
  if (score >= 40) return "s-mid";
  return "s-bad";
}

export function StatusPill({ status, score }) {
  if (!status) return <span className="pill pill-grey">Unreviewed · Check</span>;
  const ok = status === "Verified";
  return (
    <span className={`pill ${ok ? "pill-green" : "pill-red"}`}>
      {ok ? "✓ Verified" : "! Disputed"} {score}
    </span>
  );
}

export function Meter({ label, score, sub, note }) {
  return (
    <div className="meter">
      <div className="meter-head">
        <span className="meter-label">{label}</span>
        <span className={`meter-score ${scoreClass(score)}`}>
          {score}/100 <small>{sub}</small>
        </span>
      </div>
      <div className="meter-track">
        <div className={`meter-fill ${scoreClass(score)}`} style={{ width: `${Math.max(2, score)}%` }} />
      </div>
      {note && <div className="meter-note">{note}</div>}
    </div>
  );
}

export function Spinner({ text = "Working…" }) {
  return (
    <div className="spinner-row">
      <span className="spinner" /> {text}
    </div>
  );
}

export function ErrorBox({ error }) {
  if (!error) return null;
  return <div className="error-box">⚠ {String(error)}</div>;
}

export function DemoTag({ demo }) {
  return demo ? <span className="demo-tag" title="Synthetic demo article shipped with the repository">DEMO SEED</span> : null;
}

export function fmt(v, digits = 3) {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(digits);
  return String(v);
}
