import React from "react";
import { fmt } from "./ui.jsx";

export default function TelemetryStrip({ metrics }) {
  if (!metrics?.length) return null;
  return (
    <section className="card telemetry">
      <h3>📈 Executing models telemetry</h3>
      <div className="telemetry-grid">
        {metrics.map((m) => (
          <div className="tele" key={m.key}>
            <div className="tele-name">{m.model}</div>
            <div className="tele-backend muted small">{m.backend}</div>
            <div className="tele-row">
              <span>latency</span>
              <b>{m.latency_ms === null || m.latency_ms === undefined ? "—" : `${fmt(m.latency_ms, 1)} ms`}</b>
            </div>
            <div className="tele-row">
              <span>confidence</span>
              <b>{fmt(m.confidence, 2)}</b>
            </div>
            <div className="tele-bench">
              {Object.entries(m.benchmark || {}).map(([k, v]) => (
                <span key={k} className="chip">
                  {k} {fmt(v, 3)}
                </span>
              ))}
              {!Object.keys(m.benchmark || {}).length && <span className="muted small">run the evaluation suite for benchmarks</span>}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
