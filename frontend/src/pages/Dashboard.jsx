function getVerdict(result) {
  return (
    result?.final?.verdict ||
    result?.layer3?.final_verdict ||
    result?.final_verdict ||
    "needs_verification"
  );
}

function getRisk(result) {
  return (
    result?.final?.risk_score ??
    result?.layer3?.risk_score ??
    result?.final_risk_score ??
    0
  );
}

function getLabel(verdict) {
  if (
    verdict === "phishing_threat" ||
    verdict === "phishing"
  ) {
    return "Dangerous";
  }

  if (
    verdict === "ai_assisted_phishing" ||
    verdict === "ai_phish"
  ) {
    return "Dangerous";
  }

  if (
    verdict === "needs_verification" ||
    verdict === "suspicious"
  ) {
    return "Be careful";
  }

  return "Looks safe";
}

function getClass(verdict) {
  if (
    verdict === "phishing_threat" ||
    verdict === "phishing" ||
    verdict === "ai_assisted_phishing" ||
    verdict === "ai_phish"
  ) {
    return "danger";
  }

  if (
    verdict === "needs_verification" ||
    verdict === "suspicious"
  ) {
    return "warning";
  }

  return "safe";
}

export default function Dashboard({ history }) {
  const dangerous = history.filter((item) => {
    const verdict = getVerdict(item);

    return (
      verdict === "phishing_threat" ||
      verdict === "phishing" ||
      verdict === "ai_assisted_phishing" ||
      verdict === "ai_phish"
    );
  }).length;

  const careful = history.filter((item) => {
    const verdict = getVerdict(item);

    return (
      verdict === "needs_verification" ||
      verdict === "suspicious"
    );
  }).length;

  const safe = history.filter((item) => {
    const verdict = getVerdict(item);

    return (
      verdict === "likely_legitimate" ||
      verdict === "legitimate" ||
      verdict === "ham" ||
      verdict === "clean"
    );
  }).length;

  return (
    <div className="dashboard">

      <div className="analyze-intro">
        <p className="eyebrow">YOUR RESULTS</p>

        <h1>Email history</h1>

        <p>
          A quick look at the emails you've checked during this session.
        </p>
      </div>

      <div className="stats">

        <div className="stat-card">
          <span>Emails checked</span>
          <strong>{history.length}</strong>
        </div>

        <div className="stat-card">
          <span>Dangerous</span>
          <strong>{dangerous}</strong>
        </div>

        <div className="stat-card">
          <span>Be careful</span>
          <strong>{careful}</strong>
        </div>

        <div className="stat-card">
          <span>Looks safe</span>
          <strong>{safe}</strong>
        </div>

      </div>

      <div className="dashboard-card">

        <div className="dashboard-card-header">
          <div>
            <p className="eyebrow">RECENT CHECKS</p>
            <h2>Emails you've checked</h2>
          </div>
        </div>

        {history.length === 0 ? (
          <div className="empty-state">
            <h3>No emails checked yet</h3>
            <p>
              Go to "Check an Email" and analyze your first message.
            </p>
          </div>
        ) : (
          <div className="history-list">

            {history.map((item, index) => {
              const verdict = getVerdict(item);
              const label = getLabel(verdict);
              const className = getClass(verdict);
              const risk = getRisk(item);

              const sender =
                item?.layer1?.from_address ||
                item?.layer1?.from ||
                item?.sender ||
                "Sender unavailable";

              const subject =
                item?.layer1?.subject ||
                item?.subject ||
                "No subject";

              return (
                <div className="history-row" key={index}>

                  <div className="history-info">
                    <strong>{subject}</strong>
                    <span>{sender}</span>
                  </div>

                  <div className={`history-status ${className}`}>
                    {label}
                  </div>

                  <div className="history-risk">
                    Risk {risk}
                  </div>

                </div>
              );
            })}

          </div>
        )}

      </div>
    </div>
  );
}