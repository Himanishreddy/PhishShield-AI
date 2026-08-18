// Session dashboard: rolls up everything analyzed since the app was opened.
// Nothing is stored server-side; this is an in-memory view of the session.

function Stat({ label, value, tone }) {
  return (
    <div className={`stat ${tone || ""}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function verdictLabel(v) {
  if (v === "ai_phish") return "AI phishing";
  if (v === "phishing") return "Phishing";
  if (v === "suspicious") return "Suspicious";
  return "Clean";
}

function pillClass(v) {
  if (v === "ai_phish" || v === "phishing") return "phishing";
  if (v === "suspicious") return "suspicious";
  return "clean";
}

export default function Dashboard({ history }) {
  const total = history.length;
  const aiPhish = history.filter((r) => r.final_verdict === "ai_phish").length;
  const humanPhish = history.filter((r) => r.final_verdict === "phishing").length;
  const phishing = humanPhish + aiPhish;
  const suspicious = history.filter((r) => r.final_verdict === "suspicious").length;
  const clean = history.filter((r) => r.final_verdict === "clean").length;

  const pct = (n) => (total ? (n / total) * 100 : 0);

  if (total === 0) {
    return (
      <div className="dashboard">
        <h1>Dashboard</h1>
        <div className="empty-state">
          Nothing analyzed yet. Head to <strong>Analyze</strong>, run an email,
          and it'll show up here.
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <h1>Session overview</h1>
      <p className="dash-sub">
        A summary of the {total} email{total === 1 ? "" : "s"} you've analyzed
        this session.
      </p>

      <div className="stat-row">
        <Stat label="Analyzed" value={total} />
        <Stat label="Phishing" value={phishing} tone="phishing" />
        <Stat label="AI phishing" value={aiPhish} tone="phishing" />
        <Stat label="Clean" value={clean} tone="clean" />
      </div>

      <section className="dist">
        <h3>Verdict split</h3>
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
              const layer1 = r.layer1 || {};
              return (
                <tr key={i}>
                  <td>
                    <span className={`pill ${pillClass(r.final_verdict)}`}>
                      {verdictLabel(r.final_verdict)}
                    </span>
                  </td>
                  <td className="mono">{Math.round(r.final_risk_score)}</td>
                  <td className="mono-trunc">{layer1.from_address || "—"}</td>
                  <td className="subj">{layer1.subject || "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
    </div>
  );
}