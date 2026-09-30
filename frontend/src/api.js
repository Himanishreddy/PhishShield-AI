// Talks to the PhishShield backend.
// Trailing slashes are removed so "https://x.onrender.com/" also works.
const BASE = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

// Layer 3 calls an external model, so a request can take ~10-25s. The abort
// timeout is generous enough not to cut off a slow but successful review.
const TIMEOUT_MS = 90000;

export async function analyzeEmail({ email, ensemble = false, skipLayer3 = false }) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, ensemble, skip_layer3: skipLayer3 }),
      signal: controller.signal,
    });
    if (!res.ok) {
      // Technical details are logged for developers, never shown to users.
      try {
        const body = await res.json();
        console.error("PhishShield API error", res.status, body?.detail);
      } catch {
        console.error("PhishShield API error", res.status);
      }
      if (res.status >= 500) {
        throw new Error(
          "Something went wrong on our side. Please try again, or contact your administrator."
        );
      }
      throw new Error(
        "We couldn't read this email. Please paste the complete email and try again."
      );
    }
    return await res.json();
  } catch (e) {
    if (e.name === "AbortError") {
      throw new Error("The check took too long. Please try again.");
    }
    if (e instanceof TypeError) {
      throw new Error(
        "We couldn't connect to PhishShield. Please try again, or contact your administrator."
      );
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

export async function checkHealth() {
  const res = await fetch(`${BASE}/api/health`);
  if (!res.ok) throw new Error("Backend not reachable");
  return res.json();
}