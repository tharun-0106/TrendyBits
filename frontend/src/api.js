// Thin fetch wrapper around the FastAPI backend. VITE_API_BASE lets a static build talk to a remote API.
const BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/api/health"),
  models: () => request("/api/models"),
  ticker: () => request("/api/ticker"),
  articles: (topic, q) =>
    request(`/api/articles?limit=60${topic ? `&topic=${encodeURIComponent(topic)}` : ""}${q ? `&q=${encodeURIComponent(q)}` : ""}`),
  article: (id) => request(`/api/articles/${encodeURIComponent(id)}`),
  search: (body) => request("/api/search", { method: "POST", body }),
  chat: (body) => request("/api/chat", { method: "POST", body }),
  summarize: (body) => request("/api/summarize", { method: "POST", body }),
  analyze: (body) => request("/api/analyze", { method: "POST", body }),
  recentReports: () => request("/api/reports/recent?limit=9"),
  report: (id) => request(`/api/reports/${id}`),
  scrapeRss: () => request("/api/scrape/rss", { method: "POST" }),
  scrapeUrl: (url) => request("/api/scrape/url", { method: "POST", body: { url } }),
  profile: (uid) => request(`/api/profile/${uid}`),
  saveProfile: (uid, body) => request(`/api/profile/${uid}`, { method: "PUT", body }),
  feedback: (body) => request("/api/feedback", { method: "POST", body }),
  xai: (body) => request("/api/xai/explain", { method: "POST", body }),
  adminMetrics: () => request("/api/admin/metrics"),
  runTests: (suite = "all") => request("/api/admin/run-tests", { method: "POST", body: { suite } }),
  job: (id) => request(`/api/admin/jobs/${id}`),
};
