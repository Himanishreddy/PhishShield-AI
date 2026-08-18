// Shows the pipeline's verdict: a headline banner plus a breakdown of what
// each detection layer found. Color reflects threat level.

function verdictClass(v) {
  // ai_phish is still phishing — show it with the same red treatment.
  if (v === "phishing" || v === "ai_phish") return "phishing";
  if (v === "suspicious") return "suspicious";
  return "clean";
}

function verdictLabel(v) {
  if (v === "ai_phish") return "AI-Generated Phishing";
  if (v === "phishing") return "Phishing";
  if (v === "suspicious") return "Suspicious";
  return "Clean";
}

function AuthChip({ label, value }) {
  const cls = value === "pass" ? "pass" : value === "fail" ? "fail" : "unknown";
  return (
    <div className="auth-chip">
      <span className="auth-label">{label}</span>
      <span className={`auth-value ${cls}`}>{value || "—"}</span>
    </div>
  );
}

export default function Verdict({ result }) {
  const verdict = result.final_verdict;
  const score = result.final_risk_score;
  const layer1 = result.layer1 || {};
  const layer2 = result.layer2;
  const layer3 = result.layer3;
  const auth = layer1.auth || {};

  return (
    <div className="verdict-wrap">
      <div className={`verdict-banner ${verdictClass(verdict)}`}>
        <div className="verdict-head">
          <span className="verdict-eyebrow">Final verdict · fused risk</span>
          <span className="verdict-title">
            {verdictLabel(verdict)} · {Math.round(score)}/100
          </span>
        </div>
        <div className="score-track">
          <div
            className={`score-fill ${verdictClass(verdict)}`}
            style={{ width: `${score}%` }}
          />
        </div>
        <div className="verdict-path">
          {result.layer2_ran
            ? "Rules and classifier both ran"
            : "Rules only — classifier was skipped"}
          {result.layer3_ran ? " · attribution attached" : ""}
        </div>
      </div>

      <div className="layer-grid">
        <section className="layer-card">
          <h3>Layer 1 · Infrastructure</h3>
          <div className="auth-row">
            <AuthChip label="SPF" value={auth.spf} />
            <AuthChip label="DKIM" value={auth.dkim} />
            <AuthChip label="DMARC" value={auth.dmarc} />
          </div>
          <div className="kv">
            <span className="k">Infra risk</span>
            <span className="v">{layer1.infra_risk_score}/100</span>
          </div>
          {layer1.from_address && (
            <div className="kv">
              <span className="k">From</span>
              <span className="v mono-trunc">{layer1.from_address}</span>
            </div>
          )}
          {layer1.reasons && layer1.reasons.length > 0 && (
            <ul className="reasons">
              {layer1.reasons.map((reason, i) => (
                <li key={i}>{reason}</li>
              ))}
            </ul>
          )}
        </section>

        <section className="layer-card">
          <h3>Layer 2 · Language</h3>
          {layer2 == null ? (
            <div className="skipped">
              Skipped. Layer 1 cleared this email from an authenticated sender,
              so the classifier wasn't needed.
            </div>
          ) : (
            <>
              <div className="kv">
                <span className="k">Prediction</span>
                <span className="v">{layer2.predicted_label}</span>
              </div>
              <div className="kv">
                <span className="k">Confidence</span>
                <span className="v">{(layer2.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="probs">
                {Object.entries(layer2.probabilities || {}).map(([label, p]) => (
                  <div className="prob" key={label}>
                    <div className="prob-top">
                      <span>{label}</span>
                      <span>{(p * 100).toFixed(1)}%</span>
                    </div>
                    <div className="prob-track">
                      <div className="prob-fill" style={{ width: `${p * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </section>
      </div>

      {layer3 && layer3._meta && layer3._meta.status === "ok" && (
        <section className="layer-card attribution">
          <h3>Layer 3 · Attribution</h3>
          <div className="attr-grid">
            <div className="kv">
              <span className="k">Objective</span>
              <span className="v">{layer3.primary_objective}</span>
            </div>
            <div className="kv">
              <span className="k">Target</span>
              <span className="v">{layer3.target_persona}</span>
            </div>
            <div className="kv">
              <span className="k">Sophistication</span>
              <span className="v">{layer3.sophistication}</span>
            </div>
          </div>
          {layer3.psychological_triggers && (
            <div className="triggers">
              {layer3.psychological_triggers.map((t, i) => (
                <span className="trigger" key={i}>
                  {t}
                </span>
              ))}
            </div>
          )}
          {layer3.analyst_summary && (
            <div className="analyst-summary">
              <span className="as-label">Analyst summary</span>
              {layer3.analyst_summary}
            </div>
          )}
          <div className="attr-disclaimer">{layer3._meta.disclaimer}</div>
        </section>
      )}
      {layer3 && layer3._meta && layer3._meta.status !== "ok" && (
        <section className="layer-card">
          <h3>Layer 3 · Attribution</h3>
          <div className="skipped">{layer3._meta.detail}</div>
        </section>
      )}
    </div>
  );
}