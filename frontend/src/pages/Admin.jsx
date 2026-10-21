import React, { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import ConfusionMatrix from "../components/ConfusionMatrix.jsx";
import { LimeHeatmap, ShapBars } from "../components/XaiPanel.jsx";
import { ErrorBox, Spinner, fmt } from "../components/ui.jsx";

const MODEL_FIELDS = {
  retriever: [
    ["Precision@K", "precision_at_k"], ["Recall@K", "recall_at_k"], ["MRR", "mrr"], ["Intra-List Diversity", "ild"],
    ["Unique Source Ratio", "unique_source_ratio"], ["Index latency (ms/query)", "latency_ms"], ["K", "k"], ["Queries", "n_queries"],
  ],
  summarizer: [
    ["ROUGE-1", "rouge1"], ["ROUGE-2", "rouge2"], ["ROUGE-L", "rougeL"], ["DeepEval Faithfulness", "faithfulness"],
    ["Faithfulness (TC-04..07)", "deepeval_faithfulness"], ["Entity & Quantity Preservation", "entity_quantity_preservation"],
    ["Intent Slot-Level F1", "intent_slot_f1"], ["Intent exact match", "intent_exact_match"], ["Articles", "n"],
  ],
  liar: [
    ["6-Class Accuracy", "accuracy"], ["Macro Precision", "macro_precision"], ["Macro Recall", "macro_recall"],
    ["Macro F1", "macro_f1"], ["Weighted F1", "weighted_f1"], ["ROC-AUC (OvR macro)", "roc_auc"], ["Test size", "n_test"],
  ],
  integrated: [
    ["Accuracy", "accuracy"], ["Precision", "precision"], ["Recall", "recall"], ["F1-Score", "f1"], ["ROC-AUC", "roc_auc"],
    ["Macro F1", "macro_f1"], ["Test size", "n_test"],
  ],
  reasoner: [
    ["Verdict Accuracy", "verdict_accuracy"], ["Hallucination Rate", "hallucination_rate"], ["Citation Validity", "citation_validity"],
    ["RAGAs Contextual Relevancy (proxy)", "contextual_relevancy"], ["RAGAs Answer Relevancy (proxy)", "answer_relevancy"],
    ["Prompt-Injection Resistance", "prompt_injection_resistance"], ["5-Run Output Consistency", "five_run_consistency"],
    ["Unverified handling", "unverified_handling"], ["Claims", "n_claims"],
  ],
};

const ABL_COLS = [
  ["P@5", "precision_at_5"], ["R@5", "recall_at_5"], ["MRR", "mrr"], ["ILD", "ild"], ["Uniq. src", "unique_source_ratio"],
  ["Verdict acc", "verdict_accuracy"], ["Halluc.", "hallucination_rate", true], ["Cite valid", "citation_validity"],
  ["Cred. band acc", "credibility_band_accuracy"], ["Misl. recall", "misleading_recall"], ["ROUGE-1", "rouge1"],
  ["ROUGE-2", "rouge2"], ["ROUGE-L", "rougeL"], ["Fact cov.", "factual_coverage"],
];

const DIM_LABELS = {
  source_reliability: "Source Reliability",
  rag_evidence_match: "RAG Evidence Match",
  bert_liar_6class: "BERT-LIAR 6-Class",
  bert_isot_bharat_binary: "BERT ISOT-Bharat Binary",
  framing_omission_penalty: "Framing / Omission Penalty",
};

function ModelCard({ m }) {
  const met = m.metrics || {};
  return (
    <section className="card model-card">
      <div className="model-head">
        <span className="model-num">Model {m.id}</span>
        <h3>{m.name}</h3>
        <div className="muted small">
          {m.role} · backend: <b>{m.backend}</b>
        </div>
      </div>
      {!m.evaluated && <p className="muted">Not evaluated yet — press “Re-Run”.</p>}
      <table className="kv">
        <tbody>
          {MODEL_FIELDS[m.key].map(([label, key]) =>
            met[key] === undefined ? null : (
              <tr key={key}>
                <td>{label}</td>
                <td>
                  <b>{fmt(met[key])}</b>
                </td>
              </tr>
            )
          )}
        </tbody>
      </table>
      {m.key === "liar" && met.per_class && (
        <>
          <h4>Per-class F1</h4>
          <div className="perclass">
            {Object.entries(met.per_class).map(([k, v]) => (
              <div key={k} className="pc">
                <span>{k}</span>
                <div className="pc-track">
                  <div className="pc-fill" style={{ width: `${v.f1 * 100}%` }} />
                </div>
                <b>{v.f1.toFixed(3)}</b>
              </div>
            ))}
          </div>
          <ConfusionMatrix matrix={met.confusion_matrix} labels={met.labels} title="6 × 6 confusion matrix" />
        </>
      )}
      {m.key === "integrated" && met.confusion_matrix && (
        <ConfusionMatrix matrix={met.confusion_matrix} labels={["False (0)", "True (1)"]} title="2 × 2 confusion matrix" />
      )}
      {m.key === "reasoner" && met.confusion_matrix && <ConfusionMatrix matrix={met.confusion_matrix} labels={met.labels} title="Verdict confusion (24-claim benchmark)" />}
      {m.baselines && Object.keys(m.baselines).length > 0 && (
        <details>
          <summary>Baselines ({Object.keys(m.baselines).join(", ")})</summary>
          <table className="kv">
            <tbody>
              {Object.entries(m.baselines).map(([name, b]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>
                    acc {fmt(b.accuracy)} · macro-F1 {fmt(b.macro_f1)} · AUC {fmt(b.roc_auc)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </section>
  );
}

function AblationTable({ ablation }) {
  const [focus, setFocus] = useState(null);
  if (!ablation?.rows) return <p className="muted">No ablation results yet.</p>;
  const base = ablation.rows[0];
  return (
    <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            <th>Configuration</th>
            {ABL_COLS.map(([l, k]) => (
              <th key={k} className={focus === k ? "focus" : ""} onClick={() => setFocus(focus === k ? null : k)} title="click to highlight">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ablation.rows.map((r) => (
            <tr key={r.configuration}>
              <td>
                <b>{r.configuration}</b>
                <div className="muted small">{r.description}</div>
              </td>
              {ABL_COLS.map(([, k, lowerBetter]) => {
                const d = (r[k] ?? 0) - (base[k] ?? 0);
                const worse = lowerBetter ? d > 0.0005 : d < -0.0005;
                const better = lowerBetter ? d < -0.0005 : d > 0.0005;
                return (
                  <td key={k} className={`${worse ? "worse" : better ? "better" : ""} ${focus === k ? "focus" : ""}`}>
                    {fmt(r[k])}
                    {r !== base && Math.abs(d) > 0.0005 && (
                      <span className="delta">
                        {d > 0 ? "+" : ""}
                        {d.toFixed(3)}
                      </span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="muted small">
        Backends: {Object.entries(ablation.backends || {}).map(([k, v]) => `${k}=${v}`).join(" · ")} · samples:{" "}
        {Object.entries(ablation.samples || {}).map(([k, v]) => `${k} ${v}`).join(", ")} · generated {ablation.generated_at}
      </div>
    </div>
  );
}

function TestTables({ tests }) {
  const [open, setOpen] = useState(null);
  if (!tests?.cases) return <p className="muted">No test results yet.</p>;
  return (
    <>
      <h3>Table 5 — Result-reporting template</h3>
      <table className="data">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Target</th>
            <th>Measured</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {tests.table5.map((r) => (
            <tr key={r.metric}>
              <td>{r.metric}</td>
              <td>{r.target}</td>
              <td>
                <b>{fmt(r.measured)}</b>
              </td>
              <td>
                <span className={`badge ${r.status === "PASS" ? "v-supported" : "v-false"}`}>{r.status}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <h3>
        All 18 LLM test cases — {tests.passed}/{tests.total} passed
      </h3>
      <div className="muted small">
        Engine: {Object.entries(tests.engine || {}).map(([k, v]) => `${k}=${Array.isArray(v) ? v.join("/") || "none" : v}`).join(" · ")}
      </div>
      <table className="data">
        <thead>
          <tr>
            <th>TC</th>
            <th>Level</th>
            <th>Component</th>
            <th>Scenario</th>
            <th>Metric</th>
            <th>Target</th>
            <th>Measured</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {tests.cases.map((c) => (
            <React.Fragment key={c.id}>
              <tr className="clickable" onClick={() => setOpen(open === c.id ? null : c.id)}>
                <td>
                  <b>{c.id}</b>
                </td>
                <td>{c.level}</td>
                <td>{c.component}</td>
                <td>{c.name}</td>
                <td>{c.metric}</td>
                <td>{c.target}</td>
                <td>{fmt(c.measured)}</td>
                <td>
                  <span className={`badge ${c.status === "PASS" ? "v-supported" : "v-false"}`}>{c.status}</span>
                </td>
              </tr>
              {open === c.id && (
                <tr>
                  <td colSpan={8}>
                    <pre className="details">{JSON.stringify(c.details, null, 2)}</pre>
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </>
  );
}

function XaiSandbox({ globalXai }) {
  const [text, setText] = useState(
    "SHOCKER: Government to secretly give Rs 50,000 to everyone, says anonymous ministry source. Forward immediately before it is deleted!"
  );
  const [samples, setSamples] = useState(300);
  const [lime, setLime] = useState(null);
  const [dims, setDims] = useState({
    source_reliability: 35,
    rag_evidence_match: 24,
    bert_liar_6class: 40,
    bert_isot_bharat_binary: 26,
    framing_omission_penalty: 15,
  });
  const [shap, setShap] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  useEffect(() => {
    const t = setTimeout(() => api.xai({ dims }).then((r) => setShap(r.shap)).catch((e) => setErr(e.message)), 150);
    return () => clearTimeout(t);
  }, [dims]);

  async function runLime() {
    setBusy(true);
    setErr(null);
    try {
      const r = await api.xai({ text, num_samples: samples });
      setLime(r.lime);
      if (r.dims) setDims(Object.fromEntries(Object.entries(r.dims).map(([k, v]) => [k, Math.round(v)])));
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  const maxGlobal = Math.max(1, ...(globalXai?.dimensions || []).map((d) => d.mean_abs));
  return (
    <div className="sandbox">
      <ErrorBox error={err} />
      <div className="two-col">
        <div>
          <h4>LIME perturbation sandbox</h4>
          <textarea className="input" rows={4} value={text} onChange={(e) => setText(e.target.value)} />
          <label className="field">
            Perturbation samples: {samples}
            <input type="range" min={50} max={1000} step={50} value={samples} onChange={(e) => setSamples(+e.target.value)} />
          </label>
          <button className="btn btn-primary" onClick={runLime} disabled={busy}>
            {busy ? "Perturbing…" : "Explain with LIME"}
          </button>
          {lime && <LimeHeatmap lime={lime} />}
        </div>
        <div>
          <h4>Local SHAP — drag the dimension values</h4>
          {Object.keys(dims).map((k) => (
            <label className="field slider" key={k}>
              <span>
                {DIM_LABELS[k]}: <b>{dims[k]}</b>
              </span>
              <input
                type="range"
                min={0}
                max={k === "framing_omission_penalty" ? 15 : 100}
                value={dims[k]}
                onChange={(e) => setDims({ ...dims, [k]: +e.target.value })}
              />
            </label>
          ))}
          {shap && <ShapBars shap={shap} compact />}
          <h4>Global SHAP — mean |φ| over {globalXai?.n_reports || 0} saved reports</h4>
          {(globalXai?.dimensions || []).map((d) => (
            <div className="pc" key={d.key}>
              <span>{d.label}</span>
              <div className="pc-track">
                <div className="pc-fill" style={{ width: `${(d.mean_abs / maxGlobal) * 100}%` }} />
              </div>
              <b>
                {d.mean_abs} <span className="muted small">(mean {d.mean > 0 ? "+" : ""}{d.mean})</span>
              </b>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function Admin({ onMetrics }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [job, setJob] = useState(null);
  const poll = useRef(null);

  const load = () =>
    api
      .adminMetrics()
      .then((d) => {
        setData(d);
        onMetrics?.(d);
      })
      .catch((e) => setError(e.message));
  useEffect(() => {
    load();
    return () => clearInterval(poll.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function rerun(suite) {
    try {
      const j = await api.runTests(suite);
      setJob(j);
      clearInterval(poll.current);
      poll.current = setInterval(async () => {
        const s = await api.job(j.id);
        setJob(s);
        if (s.status !== "running") {
          clearInterval(poll.current);
          load();
        }
      }, 1500);
    } catch (e) {
      setError(e.message);
    }
  }

  if (error && !data) return <ErrorBox error={error} />;
  if (!data) return <Spinner text="Loading model metrics…" />;
  return (
    <div className="admin">
      <div className="admin-head">
        <div>
          <h1>📊 Admin: Deep Learning Model Performance & XAI</h1>
          <p className="muted">
            Evaluation telemetry for all 5 executing models (Section V Table 1, Section VI Tables 2–5). Benchmarks are synthetic / demo-seed
            partitions unless real IEEE Dataport files were imported.
          </p>
        </div>
        <div className="admin-actions">
          <button className="btn btn-primary" onClick={() => rerun("all")} disabled={job?.status === "running"}>
            ▶️ Re-Run All Model Tests & Ablation Suite
          </button>
          <div className="row-gap">
            {["classifiers", "ablation", "llm"].map((s) => (
              <button key={s} className="chip chip-btn" onClick={() => rerun(s)} disabled={job?.status === "running"}>
                only {s}
              </button>
            ))}
          </div>
        </div>
      </div>
      {job && (
        <div className={`card job job-${job.status}`}>
          <b>
            Job {job.id} ({job.suite}): {job.status}
          </b>
          {job.error && <div className="error-box">{job.error}</div>}
          <pre className="details">{job.log.slice(-12).join("\n")}</pre>
        </div>
      )}
      <div className="models-grid">
        {data.models.map((m) => (
          <ModelCard key={m.key} m={m} />
        ))}
      </div>
      <section className="card">
        <h2>Table 1 — Ablation study (Models 1–6)</h2>
        <p className="muted small">Red = worse than the complete system, green = better. Click a column header to highlight it.</p>
        <AblationTable ablation={data.ablation} />
      </section>
      <section className="card">
        <h2>Section VI — LLM test suite</h2>
        <TestTables tests={data.llm_tests} />
      </section>
      <section className="card">
        <h2>🧠 Interactive XAI sandbox</h2>
        <XaiSandbox globalXai={data.xai_global} />
      </section>
      <section className="card">
        <h2>Security events & feedback</h2>
        <div className="two-col">
          <div>
            <h4>Recent prompt-injection / safety flags</h4>
            <ul className="small">
              {(data.security_events || []).slice(0, 10).map((e) => (
                <li key={e.id}>
                  <span className="chip chip-warn">{e.kind}</span> {e.origin} · {e.created_at}
                </li>
              ))}
              {!data.security_events?.length && <li className="muted">none</li>}
            </ul>
          </div>
          <div>
            <h4>User feedback ratings</h4>
            <ul className="small">
              {Object.entries(data.feedback || {}).map(([k, v]) => (
                <li key={k}>
                  {k}: {v.avg_rating}★ ({v.count})
                </li>
              ))}
              {!Object.keys(data.feedback || {}).length && <li className="muted">no ratings yet</li>}
            </ul>
            <h4>Evaluation history</h4>
            <ul className="small">
              {(data.eval_history || []).slice(0, 6).map((e) => (
                <li key={e.id}>
                  {e.suite} · {e.created_at}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
