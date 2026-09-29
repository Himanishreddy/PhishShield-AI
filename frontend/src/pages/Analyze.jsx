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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [fileName, setFileName] = useState("");

  const run = async () => {
    setError(null);
    setResult(null);
    setLoading(true);

    try {
      const response = await analyzeEmail({ email });

      setResult(response);
      onResult(response);
    } catch (e) {
      setError(e.message || "Something went wrong while checking the email.");
    } finally {
      setLoading(false);
    }
  };

  const handleFile = (e) => {
    const file = e.target.files?.[0];

    if (!file) return;

    setFileName(file.name);
    setError(null);

    const reader = new FileReader();

    reader.onload = (event) => {
      setEmail(event.target?.result || "");
    };

    reader.onerror = () => {
      setError("We couldn't read that email file.");
    };

    reader.readAsText(file);
  };

  return (
    <div className="analyze">

      <div className="analyze-intro">
        <p className="eyebrow">EMAIL SECURITY</p>

        <h1>Is this email safe?</h1>

        <p>
          Paste an email below or upload an <code>.eml</code> file.
          We'll check the sender, links, and message content for signs of
          a scam.
        </p>
      </div>

      <div className="email-box">

        <textarea
          className="email-input"
          value={email}
          onChange={(e) => {
            setEmail(e.target.value);
            setFileName("");
          }}
          spellCheck={false}
          placeholder="Paste the complete email here..."
        />

        <div className="email-actions">

          <label className="file-upload-label">
            <input
              type="file"
              accept=".eml,message/rfc822"
              onChange={handleFile}
            />

            <span>Open an .eml file</span>
          </label>

          <button
            type="button"
            className="text-button"
            onClick={() => setEmail(SAMPLE)}
          >
            Use an example
          </button>

          <button
            type="button"
            className="text-button"
            onClick={() => {
              setEmail("");
              setFileName("");
              setResult(null);
              setError(null);
            }}
          >
            Clear
          </button>

          <button
            className="analyze-btn"
            onClick={run}
            disabled={loading || !email.trim()}
          >
            {loading ? "Checking..." : "Check this email"}
          </button>
        </div>

        {fileName && (
          <div className="selected-file">
            Selected: <strong>{fileName}</strong>
          </div>
        )}
      </div>

      {error && (
        <div className="error-panel">
          <strong>We couldn't check this email.</strong>
          <p>{error}</p>
          <div className="error-hint">
            If this keeps happening, contact your administrator.
          </div>
        </div>
      )}

      {loading && (
        <div className="loading-panel">
          <strong>Checking your email...</strong>

          <p>
            We're checking the sender, links, and message content.
            This can take a few seconds.
          </p>
        </div>
      )}

      {result && <Verdict result={result} />}
    </div>
  );
}