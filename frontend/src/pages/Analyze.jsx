import { useState } from "react";
import { analyzeEmail } from "../api";
import Verdict from "../components/Verdict";

const SAMPLE = `From: "Microsoft Support" <security-update@micros0ft-support.com>
Reply-To: attacker-collect@totally-diff-domain.ru
To: cfo@yourcompany.com
Subject: URGENT: Action Required Immediately - Account Suspended
Authentication-Results: mx.company.com; spf=fail; dkim=fail; dmarc=fail
Content-Type: text/html

Your account will be locked within 2 hours due to unauthorized login attempt.
Click here to verify your identity: http://secure-login-portal.xyz/verify`;

export default function Analyze({ onResult }) {
  const [email, setEmail] = useState(SAMPLE);
  const [ensemble, setEnsemble] = useState(false);
  const [runLayer3, setRunLayer3] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const run = async () => {
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const r = await analyzeEmail({ email, ensemble, runLayer3 });
      setResult(r);
      onResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="analyze">
      <div className="analyze-intro">
        <h1>Analyze an email</h1>
        <p>
          Paste a message with its headers. It runs through the rules filter and
          the classifier; confirmed threats get an attribution pass.
        </p>
      </div>

      <textarea
        className="email-input"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        spellCheck={false}
        placeholder="Paste raw email here, headers included…"
      />

      <div className="controls">
        <div className="toggles">
          <label className="toggle">
            <input
              type="checkbox"
              checked={ensemble}
              onChange={(e) => setEnsemble(e.target.checked)}
            />
            <span>Run classifier on every email</span>
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={runLayer3}
              onChange={(e) => setRunLayer3(e.target.checked)}
            />
            <span>Attribution on confirmed phishing</span>
          </label>
        </div>
        <button className="analyze-btn" onClick={run} disabled={loading || !email.trim()}>
          {loading ? "Analyzing…" : "Analyze email"}
        </button>
      </div>

      {error && (
        <div className="error-panel">
          <strong>Couldn't analyze.</strong> {error}
          <div className="error-hint">
            Make sure the backend is running: <code>python -m uvicorn backend.main:app --port 8000</code>
          </div>
        </div>
      )}

      {loading && (
        <div className="loading-panel">
          Scoring the message. The first run loads the model into memory and can
          take a few seconds.
        </div>
      )}

      {result && <Verdict result={result} />}
    </div>
  );
}