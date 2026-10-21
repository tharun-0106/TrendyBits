import React from "react";

export default function ConfusionMatrix({ matrix, labels, title }) {
  if (!matrix?.length) return null;
  const max = Math.max(1, ...matrix.flat());
  return (
    <div className="cm">
      {title && <div className="cm-title">{title}</div>}
      <table>
        <thead>
          <tr>
            <th className="muted small">gold ↓ / pred →</th>
            {labels.map((l) => (
              <th key={l}>{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th>{labels[i]}</th>
              {row.map((v, j) => (
                <td
                  key={j}
                  style={{ background: `rgba(${i === j ? "22,163,74" : "200,16,46"}, ${0.08 + (0.75 * v) / max})` }}
                  title={`gold ${labels[i]} → predicted ${labels[j]}: ${v}`}
                >
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
