function verdictClass(v) {
  if (v === "phishing") return "phishing";
  if (v === "suspicious") return "suspicious";
  return "clean";
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
  const v = result.final_verdict;
  const score = result.final_risk_score;
  const l1 = result.layer1 || {};
  const l2 = result.layer2;
  const l3 = result.layer3;
  const auth = l1.auth || {};

  return (
    <div className="verdict-wrap">
      <div className={`verdict-banner ${verdictClass(v)}`}>
        <div className="verdict-head">
          <span className="verdict-eyebrow">Final verdict · fused risk</span>
          <span className="verdict-title">
            {v.toUpperCase()} · {Math.round(score)}/100
          </span>
        </div>
        <div className="score-track">
          <div className={`score-fill ${verdictClass(v)}`} style={{ width: `${score}%` }} />
        </div>
        <div className="verdict-path">
          {result.layer2_ran ? "Rules + classifier ran" : "Rules only (classifier gated out)"}
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
            <span className="k">infra risk</span>
            <span className="v">{l1.infra_risk_score}/100</span>
          </div>
          {l1.from_address && (
            <div className="kv">
              <span className="k">from</span>
              <span className="v mono-trunc">{l1.from_address}</span>
            </div>
          )}
          {l1.reasons && l1.reasons.length > 0 && (
            <ul className="reasons">
              {l1.reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          )}
        </section>

        <section className="layer-card">
          <h3>Layer 2 · Semantic</h3>
          {l2 == null ? (
            <div className="skipped">
              Skipped — Layer 1 judged this clean from an authenticated sender, so
              the classifier wasn't needed.
            </div>
          ) : (
            <>
              <div className="kv">
                <span className="k">predicted</span>
                <span className="v">{l2.predicted_label}</span>
              </div>
              <div className="kv">
                <span className="k">confidence</span>
                <span className="v">{(l2.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="probs">
                {Object.entries(l2.probabilities || {}).map(([label, p]) => (
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

      {l3 && l3._meta && l3._meta.status === "ok" && (
        <section className="layer-card attribution">
          <h3>Layer 3 · Attribution (inferred)</h3>
          <div className="attr-grid">
            <div className="kv">
              <span className="k">objective</span>
              <span className="v">{l3.primary_objective}</span>
            </div>
            <div className="kv">
              <span className="k">target</span>
              <span className="v">{l3.target_persona}</span>
            </div>
            <div className="kv">
              <span className="k">sophistication</span>
              <span className="v">{l3.sophistication}</span>
            </div>
          </div>
          {l3.psychological_triggers && (
            <div className="triggers">
              {l3.psychological_triggers.map((t, i) => (
                <span className="trigger" key={i}>
                  {t}
                </span>
              ))}
            </div>
          )}
          {l3.analyst_summary && (
            <div className="analyst-summary">
              <span className="as-label">Analyst summary</span>
              {l3.analyst_summary}
            </div>
          )}
          <div className="attr-disclaimer">{l3._meta.disclaimer}</div>
        </section>
      )}
      {l3 && l3._meta && l3._meta.status !== "ok" && (
        <section className="layer-card">
          <h3>Layer 3 · Attribution</h3>
          <div className="skipped">{l3._meta.detail}</div>
        </section>
      )}
    </div>
  );
}