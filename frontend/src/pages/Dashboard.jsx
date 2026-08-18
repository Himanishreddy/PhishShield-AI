function Stat({ label, value, tone }) {
  return (
    <div className={`stat ${tone || ""}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

export default function Dashboard({ history }) {
  const total = history.length;
  const phishing = history.filter((r) => r.final_verdict === "phishing").length;
  const suspicious = history.filter((r) => r.final_verdict === "suspicious").length;
  const clean = history.filter((r) => r.final_verdict === "clean").length;

  const pct = (n) => (total ? (n / total) * 100 : 0);

  if (total === 0) {
    return (
      <div className="dashboard empty">
        <h1>Dashboard</h1>
        <div className="empty-state">
          Nothing analyzed yet this session. Head to <strong>Analyze</strong>,
          run an email, and its result will roll up here.
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <h1>Session overview</h1>
      <p className="dash-sub">
        Summary of the {total} email{total === 1 ? "" : "s"} analyzed since you
        opened the app.
      </p>

      <div className="stat-row">
        <Stat label="Analyzed" value={total} />
        <Stat label="Phishing" value={phishing} tone="phishing" />
        <Stat label="Suspicious" value={suspicious} tone="suspicious" />
        <Stat label="Clean" value={clean} tone="clean" />
      </div>

      <section className="dist">
        <h3>Verdict distribution</h3>
        <div className="dist-bar">
          <div className="dist-seg phishing" style={{ width: `${pct(phishing)}%` }} />
          <div className="dist-seg suspicious" style={{ width: `${pct(suspicious)}%` }} />
          <div className="dist-seg clean" style={{ width: `${pct(clean)}%` }} />
        </div>
        <div className="dist-legend">
          <span><i className="dot phishing" /> Phishing {phishing}</span>
          <span><i className="dot suspicious" /> Suspicious {suspicious}</span>
          <span><i className="dot clean" /> Clean {clean}</span>
        </div>
      </section>

      <section className="recent">
        <h3>Recent analyses</h3>
        <table className="recent-table">
          <thead>
            <tr>
              <th>Verdict</th>
              <th>Risk</th>
              <th>Sender</th>
              <th>Subject</th>
            </tr>
          </thead>
          <tbody>
            {history.slice(0, 12).map((r, i) => {
              const l1 = r.layer1 || {};
              return (
                <tr key={i}>
                  <td>
                    <span className={`pill ${r.final_verdict}`}>{r.final_verdict}</span>
                  </td>
                  <td className="mono">{Math.round(r.final_risk_score)}</td>
                  <td className="mono-trunc">{l1.from_address || "—"}</td>
                  <td className="subj">{l1.subject || "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
    </div>
  );
}