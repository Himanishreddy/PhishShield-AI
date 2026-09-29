// Result card shown after an email is checked.
// Everything on this card is written for ordinary people: no technical terms
// (SPF, DKIM, DMARC, Reply-To, layers, models). Technical details stay in the
// API response for developers; they are never shown here.

const UNAVAILABLE_MESSAGE =
  "A detailed description isn't available right now. Please contact your administrator.";

function getFinal(result) {
  return result?.final || result?.layer3 || {};
}

function isDegraded(result) {
  return Boolean(getFinal(result)?.degraded);
}

function isFromReview(result) {
  return getFinal(result)?.decided_by === "layer3";
}

function getVerdict(result) {
  const final = getFinal(result);

  return (
    final?.verdict ||
    result?.final_verdict ||
    "needs_verification"
  );
}

function getRisk(result) {
  const final = getFinal(result);

  return (
    final?.risk_score ??
    result?.final_risk_score ??
    result?.risk_score ??
    0
  );
}

function getConfidence(result) {
  if (!isFromReview(result)) return null;
  const value = getFinal(result)?.confidence;
  return typeof value === "number" ? value : null;
}

function getExplanation(result) {
  if (isDegraded(result)) {
    return UNAVAILABLE_MESSAGE;
  }

  const final = getFinal(result);

  if (isFromReview(result) && final?.debrief) {
    return final.debrief;
  }

  return "We checked who sent this email, where its links go, and what the message says.";
}

function getKeyPoints(result) {
  if (!isFromReview(result)) return [];
  const evidence = getFinal(result)?.evidence;
  return Array.isArray(evidence) ? evidence.filter(Boolean) : [];
}

function getAction(verdict) {
  if (verdict === "phishing_threat" || verdict === "phishing") {
    return "Don't click links, open attachments, reply, or share personal information. If the email claims to be from an organization, contact them through their official website or phone number.";
  }

  if (verdict === "ai_assisted_phishing" || verdict === "ai_phish") {
    return "Treat this message as a scam. Don't click links or share any information until you have checked with the sender through a trusted channel.";
  }

  if (verdict === "needs_verification" || verdict === "suspicious") {
    return "Be careful before acting. Check with the sender through a trusted channel before you pay, log in, or share personal information.";
  }

  return "Nothing suggests a scam. Still use normal care with unexpected links, attachments, and requests for personal information.";
}

function getRecommended(result, verdict) {
  const final = getFinal(result);

  if (isFromReview(result) && final?.recommended_action) {
    return final.recommended_action;
  }

  return getAction(verdict);
}

function getPresentation(verdict) {
  if (verdict === "phishing_threat" || verdict === "phishing") {
    return {
      className: "danger",
      title: "Dangerous email",
      subtitle: "This message shows strong signs of a scam."
    };
  }

  if (verdict === "ai_assisted_phishing" || verdict === "ai_phish") {
    return {
      className: "danger",
      title: "Likely scam email",
      subtitle: "This message shows strong signs of a scam."
    };
  }

  if (verdict === "needs_verification" || verdict === "suspicious") {
    return {
      className: "warning",
      title: "Be careful",
      subtitle: "We found some things that deserve a closer look."
    };
  }

  return {
    className: "safe",
    title: "Looks safe",
    subtitle: "Nothing in this message suggests a scam."
  };
}

// Turn the automatic rule-based warnings into everyday language.
function toPlainWarning(reason) {
  const text = String(reason || "");
  const lower = text.toLowerCase();

  if (/(dmarc|spf|dkim) fail/.test(lower)) {
    return "We could not confirm that this email really came from the sender it shows.";
  }
  if (lower.includes("no dmarc policy")) {
    return "The sender's email address has weak protection against being faked.";
  }
  if (lower.startsWith("lookalike domain")) {
    const brand = text.match(/of '([^']+)'/);
    return brand
      ? `The sender's address imitates a well-known name (${brand[1]}).`
      : "The sender's address imitates a well-known name.";
  }
  if (lower.startsWith("display name impersonates")) {
    return "The sender's name claims to be a well-known company, but the email address doesn't match.";
  }
  if (lower.startsWith("reply-to domain differs")) {
    return "Replies to this email would go to a different address than the one it came from.";
  }
  if (lower.startsWith("embedded links point")) {
    return "Links in the email lead to other websites than the sender's own.";
  }
  if (lower.startsWith("urgency")) {
    return "The subject line uses pressure words, such as 'urgent'.";
  }
  if (lower.startsWith("sender domain registered")) {
    return "The sender's web address was created very recently.";
  }
  return null;
}

function getWarnings(result) {
  const reasons = result?.layer1?.reasons;
  if (!Array.isArray(reasons)) return [];

  const seen = new Set();
  const out = [];
  for (const reason of reasons) {
    const plain = toPlainWarning(reason);
    if (plain && !seen.has(plain)) {
      seen.add(plain);
      out.push(plain);
    }
  }
  return out;
}

function getMessageSummary(result) {
  const label = result?.layer2?.predicted_label;

  if (label === "ham") {
    return "The wording reads like a normal email.";
  }
  if (label === "phishing") {
    return "The wording looks similar to known scam emails.";
  }
  if (label === "ai_phish") {
    return "The wording looks similar to known scam emails, including ones written with AI tools.";
  }
  return "The wording was not checked separately for this email.";
}

export default function Verdict({ result }) {
  const verdict = getVerdict(result);
  const risk = getRisk(result);
  const confidence = getConfidence(result);
  const explanation = getExplanation(result);
  const presentation = getPresentation(verdict);
  const keyPoints = getKeyPoints(result);
  const recommended = getRecommended(result, verdict);
  const warnings = getWarnings(result);
  const degraded = isDegraded(result);

  return (
    <section className={`verdict ${presentation.className}`}>

      <div className="verdict-line" />

      <div className="verdict-header">
        <p className="eyebrow">EMAIL SAFETY RESULT</p>

        <h2>{presentation.title}</h2>

        <p className="verdict-subtitle">
          {presentation.subtitle}
        </p>
      </div>

      <div className="verdict-grid">

        <div className="verdict-main">

          <div className="result-section">
            <h3>What we found</h3>

            <p className={degraded ? "unavailable-text" : undefined}>
              {explanation}
            </p>
          </div>

          {keyPoints.length > 0 && (
            <div className="result-section">
              <h3>Key points</h3>
              <ul>
                {keyPoints.map((point, index) => (
                  <li key={index}>{point}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="action-box">
            <h3>What you should do</h3>
            <p>{recommended}</p>
          </div>

        </div>

        <div className="result-score">

          <div className="score-number">
            {risk}
          </div>

          <div className="score-label">
            Risk (out of 100)
          </div>

          {confidence !== null && (
            <div className="confidence-text">
              Confidence: {confidence}%
            </div>
          )}

        </div>
      </div>

      <details className="details-panel">
        <summary>See more details</summary>

        <div className="details-content">

          <div className="detail-card">
            <h3>Automatic warnings</h3>

            {warnings.length > 0 ? (
              <>
                {!degraded && keyPoints.length > 0 && (
                  <p>
                    These are automatic warnings. They were already taken into
                    account above, so they don't always mean there is a problem.
                  </p>
                )}
                <ul>
                  {warnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              </>
            ) : (
              <p>No automatic warnings about the sender or links.</p>
            )}
          </div>

          <div className="detail-card">
            <h3>Message wording</h3>
            <p>{getMessageSummary(result)}</p>
          </div>

        </div>
      </details>

    </section>
  );
}
