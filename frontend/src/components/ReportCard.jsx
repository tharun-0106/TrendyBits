import React from "react";
import XaiPanel from "./XaiPanel.jsx";
import TelemetryStrip from "./TelemetryStrip.jsx";
import { Meter, VerdictBadge, scoreClass } from "./ui.jsx";

const ORDER = { FALSE: 0, MISLEADING: 1, UNVERIFIED: 2, SUPPORTED: 3 };

export function ClaimList({ claims, compact = false }) {
  const sorted = [...(claims || [])].sort((a, b) => ORDER[a.verdict] - ORDER[b.verdict]);
  return (
    <div className="claims">
      {sorted.map((c) => (
        <div className={`claim claim-${c.verdict.toLowerCase()}`} id={`claim-${c.cid}`} key={c.cid}>
          <div className="claim-head">
            <VerdictBadge verdict={c.verdict} />
            <span className="claim-text">“{c.text}”</span>
          </div>
          <div className="claim-why">
            {c.rationale}
            {c.conflict && <span className="chip chip-warn">conflict · uncertainty {c.uncertainty}</span>}
          </div>
          {!compact && c.evidence?.length > 0 && (
            <div className="claim-ev">
              Evidence:{" "}
              {c.evidence.map((e, i) => (
                <span key={e.url}>
                  {i > 0 && " • "}
                  <a href={e.url} target="_blank" rel="noreferrer" title={e.passage}>
                    {e.publisher} ({e.published_at})
                  </a>
                  <span className="muted small"> [{e.stance}, trust {Math.round(e.trust)}]</span>
                </span>
              ))}
            </div>
          )}
          {!compact && !c.evidence?.length && <div className="claim-ev muted">No sources cited — nothing relevant was retrieved.</div>}
        </div>
      ))}
    </div>
  );
}

export function ScoreMeters({ report }) {
  return (
    <div className="meters">
      <Meter label="Source Trust" score={report.source_trust.score} sub={`- ${report.source_trust.label}`} note={report.source_trust.summary} />
      <Meter label="Fact Accuracy" score={report.fact_accuracy.score} sub={`- ${report.fact_accuracy.label}`} note={report.fact_accuracy.summary} />
    </div>
  );
}

export function OverallBadge({ report }) {
  return (
    <span className={`overall ${scoreClass(report.overall_score)}`} title={report.overall_note}>
      {report.overall_display}
    </span>
  );
}

/** Full embedded Pipeline B report (article page / "Open report"). */
export default function ReportCard({ report, related, onOpenArticle }) {
  if (!report) return null;
  return (
    <section className="card report-card" id="report">
      <div className="report-head">
        <div>
          <div className="eyebrow">Pipeline B · Independent credibility report</div>
          <h2>{report.title}</h2>
          <div className="muted small">
            Reasoning model: {report.reasoning_model} · engine: {report.reasoner_used?.join(", ")} · {report.latency_ms} ms
          </div>
        </div>
        <OverallBadge report={report} />
      </div>
      <p className="muted small">{report.overall_note}</p>
      <ScoreMeters report={report} />
      {report.security_flags?.length > 0 && (
        <div className="security-note">🛡 Embedded instructions detected and ignored: {report.security_flags.join(", ")}</div>
      )}
      <h3>Tagged claims</h3>
      <ClaimList claims={report.claim_results} />
      {report.non_claims?.length > 0 && (
        <details className="non-claims">
          <summary>{report.non_claims.length} sentences not fact-checked (opinion / call-to-action / background)</summary>
          <ul>
            {report.non_claims.map((n, i) => (
              <li key={i}>
                <span className="chip">{n.type}</span> {n.text}
              </li>
            ))}
          </ul>
        </details>
      )}
      <XaiPanel xai={report.xai_explanation} />
      <TelemetryStrip metrics={report.model_execution_metrics} />
      <h3>Sources cited</h3>
      <div className="source-cards">
        {report.sources_cited?.length ? (
          report.sources_cited.map((s) => (
            <a className="source-card" href={s.url} target="_blank" rel="noreferrer" key={s.url}>
              <b>{s.title}</b>
              <span className="muted small">
                {s.publisher} · {s.published_at} · trust {Math.round(s.trust)}
              </span>
            </a>
          ))
        ) : (
          <span className="muted">No sources cited.</span>
        )}
      </div>
      <div className="two-col">
        <div>
          <h3>Key highlights</h3>
          <ul className="highlights">
            {report.highlights.map((h, i) => (
              <li key={i}>{h}</li>
            ))}
          </ul>
        </div>
        <div>
          <h3>Timeline of claim</h3>
          <ol className="timeline">
            {report.timeline.map((t, i) => (
              <li key={i}>
                <span className="tl-date">{t.date}</span> {t.event}
              </li>
            ))}
          </ol>
        </div>
      </div>
      {related?.length > 0 && (
        <>
          <h3>Related stories</h3>
          <div className="related">
            {related.map((r) => (
              <button className={`related-card st-${(r.status || "Unreviewed").toLowerCase()}`} key={r.id} onClick={() => onOpenArticle?.(r.id)}>
                <span className="related-status">
                  {r.status === "Verified" ? `Verified - ${r.score}` : r.status === "Disputed" ? `Disputed - ${r.score}` : "Unreviewed - Check"}
                </span>
                <b>{r.title}</b>
                <span className="muted small">{r.source}</span>
              </button>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
