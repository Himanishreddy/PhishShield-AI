// Central place for talking to the PhishShield FastAPI backend.
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function analyzeEmail({ email, ensemble = false, runLayer3 = false }) {
  const res = await fetch(`${BASE}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, ensemble, run_layer3: runLayer3 }),
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch (_) {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${BASE}/api/health`);
  if (!res.ok) throw new Error("Backend not reachable");
  return res.json();
}