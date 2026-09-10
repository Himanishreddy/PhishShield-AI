"""
PhishShield AI — SOC Analyst Dashboard

A Security Operations Center triage console. Paste an email (or load a .eml),
and watch it flow through Layer 1 (rules) and Layer 2 (DistilBERT), then read
the fused verdict with every piece of evidence laid out for the analyst.

Run:
    pip install streamlit
    streamlit run soc_dashboard.py

The app imports your existing Layer 1 + Layer 2 code and the pipeline
orchestrator — it does not reimplement detection, so what you see here is
exactly what the pipeline produces.

Expected layout (same as pipeline.py):
    Phishing/
      pipeline.py
      soc_dashboard.py         <- this file, in the project root
      Layer-1/layer1_detector.py
      Layer-2/predict.py
      Layer-2/models/phishing-model-3class/   (or phishing-model)
"""

from __future__ import annotations

import importlib.util
import sys
from email import message_from_string
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config + design system
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PhishShield SOC",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The visual language here is a security analyst's console: monospace data,
# precise hairlines, a restrained slate palette, and ONE signal color that
# shifts with threat level (calm cyan -> amber -> alert red). The color IS
# the information — an analyst should read the room from across it.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
  --bg: #0e1116;
  --panel: #161b22;
  --panel-2: #1c232d;
  --line: #2a333f;
  --text: #e6edf3;
  --muted: #8b98a5;
  --cyan: #39c5cf;
  --amber: #d9a441;
  --red: #f0506e;
  --green: #3fb950;
}

.stApp { background: var(--bg); }
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }code, pre, .mono { font-family: 'IBM Plex Mono', monospace; }

.block-container { padding-top: 2.2rem; max-width: 1200px; }

.ps-masthead {
  display: flex; align-items: baseline; gap: 0.9rem;
  border-bottom: 1px solid var(--line); padding-bottom: 0.9rem; margin-bottom: 1.4rem;
}
.ps-masthead h1 {
  font-family: 'IBM Plex Mono', monospace; font-weight: 600;
  font-size: 1.35rem; letter-spacing: -0.01em; color: var(--text); margin: 0;
}
.ps-masthead .tag {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
  color: var(--muted); text-transform: uppercase; letter-spacing: 0.14em;
}

/* Verdict banner — the signature element. Its color and left rule carry the
   threat level so an analyst reads it instantly. */
.verdict {
  border: 1px solid var(--line); border-left-width: 4px;
  border-radius: 8px; padding: 1.2rem 1.4rem; background: var(--panel);
  margin-bottom: 1.2rem;
}
.verdict.phishing { border-left-color: var(--red); }
.verdict.suspicious { border-left-color: var(--amber); }
.verdict.clean { border-left-color: var(--green); }
.verdict .label {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
}
.verdict .value {
  font-family: 'IBM Plex Mono', monospace; font-weight: 600;
  font-size: 1.9rem; letter-spacing: -0.01em; margin-top: 0.15rem;
}
.verdict.phishing .value { color: var(--red); }
.verdict.suspicious .value { color: var(--amber); }
.verdict.clean .value { color: var(--green); }

.score-track {
  height: 8px; background: var(--panel-2); border-radius: 99px;
  overflow: hidden; margin-top: 0.9rem;
}
.score-fill { height: 100%; border-radius: 99px; }

.panel {
  border: 1px solid var(--line); border-radius: 8px;
  background: var(--panel); padding: 1.1rem 1.2rem; height: 100%;
}
.panel h3 {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--cyan);
  margin: 0 0 0.9rem 0; font-weight: 600;
}
.kv { display: flex; justify-content: space-between; gap: 1rem;
  padding: 0.32rem 0; border-bottom: 1px dotted var(--line); font-size: 0.9rem; }
.kv:last-child { border-bottom: none; }
.kv .k { color: var(--muted); font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }
.kv .v { color: var(--text); font-family: 'IBM Plex Mono', monospace; text-align: right; }
.v.pass { color: var(--green); }
.v.fail { color: var(--red); }

.reason {
  font-size: 0.86rem; color: var(--text); padding: 0.4rem 0 0.4rem 1.1rem;
  position: relative; border-bottom: 1px dotted var(--line);
}
.reason:before { content: "▸"; position: absolute; left: 0; color: var(--red); }
.reason:last-child { border-bottom: none; }

.prob-row { margin: 0.5rem 0; }
.prob-row .plabel {
  display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace;
  font-size: 0.82rem; color: var(--text); margin-bottom: 0.25rem;
}
.prob-track { height: 6px; background: var(--panel-2); border-radius: 99px; overflow: hidden; }
.prob-fill { height: 100%; background: var(--cyan); border-radius: 99px; }

.gate-note {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: var(--muted);
  border: 1px dashed var(--line); border-radius: 6px; padding: 0.6rem 0.8rem; margin-top: 0.6rem;
}
.stTextArea textarea {
  font-family: 'IBM Plex Mono', monospace !important; font-size: 0.85rem !important;
  background: var(--panel) !important; color: var(--text) !important; border-color: var(--line) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load pipeline + layers (cached so the model loads once per session)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner="Loading detection model…")
def load_system(model_path: str):
    layer1_mod = _load_module("layer1_detector", ROOT / "Layer-1" / "layer1_detector.py")
    predict_mod = _load_module("predict", ROOT / "Layer-2" / "predict.py")
    pipeline_mod = _load_module("pipeline", ROOT / "pipeline.py")
    classifier = predict_mod.PhishClassifier(model_path)
    # Layer 3 is optional — only load it if the file exists
    layer3_mod = None
    l3_path = ROOT / "Layer-3" / "layer3_attribution.py"
    if l3_path.exists():
        layer3_mod = _load_module("layer3_attribution", l3_path)
    return layer1_mod, predict_mod, pipeline_mod, classifier, layer3_mod



# ---------------------------------------------------------------------------
# Clean PhishShield AI product interface
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg:#0b1020; --card:#11182a; --card2:#151e33; --border:#26324a;
  --text:#f4f7fb; --muted:#9aa8bf; --blue:#4da3ff;
  --green:#35d07f; --yellow:#f5c451; --red:#ff5d73;
}
.stApp {
  background:radial-gradient(circle at 80% 0%,rgba(77,163,255,.10),transparent 30%),var(--bg);
  color:var(--text);
}
html,body,[class*="css"] { font-family:'Inter',sans-serif; }
.block-container { max-width:1180px; padding-top:2rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { display:none; }

.ps-header { display:flex;align-items:center;justify-content:space-between;margin-bottom:2.2rem; }
.ps-brand { display:flex;align-items:center;gap:.8rem; }
.ps-shield { width:42px;height:42px;display:flex;align-items:center;justify-content:center;
  border:1px solid #31547e;border-radius:12px;background:#101d32;font-size:1.35rem; }
.ps-title { font-size:1.45rem;font-weight:800;letter-spacing:-.03em; }
.ps-subtitle { color:var(--muted);font-size:.82rem;margin-top:.12rem; }
.ps-status { border:1px solid #245b43;background:#0d2119;color:var(--green);
  border-radius:999px;padding:.42rem .8rem;font-size:.75rem;font-weight:600; }

.ps-intro { text-align:center;margin:1.4rem 0 1.5rem; }
.ps-intro h2 { font-size:2rem;margin:0;letter-spacing:-.04em; }
.ps-intro p { color:var(--muted);margin:.55rem 0 0; }

.input-card,.result-card { background:var(--card);border:1px solid var(--border);
  border-radius:16px;padding:1.2rem;box-shadow:0 15px 45px rgba(0,0,0,.18); }
.input-label { font-weight:600;font-size:.9rem;margin-bottom:.55rem; }
.stTextArea textarea { background:#0b1222!important;color:var(--text)!important;
  border:1px solid var(--border)!important;border-radius:10px!important;
  font-family:'Inter',sans-serif!important;font-size:.88rem!important; }
.stButton button { border-radius:9px!important;font-weight:700!important;min-height:2.7rem; }

.result-card { margin-top:1.7rem; }
.verdict { text-align:center;padding:1.2rem;border-radius:14px;border:1px solid var(--border);margin-bottom:1.2rem; }
.verdict.clean { background:rgba(53,208,127,.07);border-color:rgba(53,208,127,.35); }
.verdict.suspicious { background:rgba(245,196,81,.07);border-color:rgba(245,196,81,.35); }
.verdict.phishing { background:rgba(255,93,115,.07);border-color:rgba(255,93,115,.35); }
.verdict-icon { font-size:2rem; }
.verdict-title { font-size:1.55rem;font-weight:800;margin-top:.35rem; }
.verdict-desc { color:var(--muted);font-size:.88rem;margin-top:.35rem; }
.risk { font-size:2.1rem;font-weight:800;margin-top:.6rem; }
.risk-track,.confidence-track { height:8px;background:#202a40;border-radius:999px;overflow:hidden;margin-top:.8rem; }
.risk-fill,.confidence-fill { height:100%;border-radius:999px; }

.info-card { background:var(--card2);border:1px solid var(--border);border-radius:13px;
  padding:1rem 1.05rem;height:100%; }
.info-title { font-size:.78rem;text-transform:uppercase;letter-spacing:.09em;
  color:var(--muted);margin-bottom:.8rem;font-weight:700; }
.info-row { display:flex;justify-content:space-between;gap:.8rem;padding:.5rem 0;
  border-bottom:1px solid rgba(38,50,74,.65);font-size:.85rem; }
.info-row:last-child { border-bottom:none; }
.good{color:var(--green);font-weight:600}.warn{color:var(--yellow);font-weight:600}
.bad{color:var(--red);font-weight:600}.neutral{color:var(--text);font-weight:500}
.reason { padding:.55rem 0;color:#dce4f0;font-size:.86rem;border-bottom:1px solid rgba(38,50,74,.65); }
.reason:last-child { border-bottom:none; }
.confidence-row { margin:.7rem 0; }
.confidence-head { display:flex;justify-content:space-between;font-size:.82rem;margin-bottom:.3rem; }
.confidence-fill { background:var(--blue); }
.footer-note { text-align:center;color:#697890;font-size:.72rem;margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

# Production model is selected automatically. Users do not see implementation details.
model_path = str(ROOT / "Layer-2" / "models" / "phishing-model-3class")
ensemble = False
run_layer3 = False

st.markdown("""
<div class="ps-header">
  <div class="ps-brand">
    <div class="ps-shield">🛡️</div>
    <div>
      <div class="ps-title">PhishShield AI</div>
      <div class="ps-subtitle">Intelligent Email Security</div>
    </div>
  </div>
  <div class="ps-status">● Protection Ready</div>
</div>
<div class="ps-intro">
  <h2>Analyze an Email</h2>
  <p>Check an email for phishing, fraud, and suspicious activity.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="input-card"><div class="input-label">Email content</div>', unsafe_allow_html=True)

SAMPLE = """From: "Microsoft Support" <security-update@micros0ft-support.com>
Reply-To: attacker-collect@totally-diff-domain.ru
To: cfo@yourcompany.com
Subject: URGENT: Action Required Immediately - Account Suspended
Authentication-Results: mx.company.com; spf=fail; dkim=fail; dmarc=fail
Content-Type: text/html

Your account will be locked within 2 hours due to unauthorized login attempt.
Click here to verify your identity: http://secure-login-portal.xyz/verify
"""

raw_email = st.text_area("Email content", value=SAMPLE, height=230, label_visibility="collapsed")

c1, c2, c3 = st.columns([1.5, 1.5, 1])
with c1:
    analyze = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
with c2:
    uploaded = st.file_uploader("Upload .eml", type=["eml"], label_visibility="collapsed")
    if uploaded is not None:
        raw_email = uploaded.read().decode(errors="ignore")
with c3:
    if st.button("Clear", use_container_width=True):
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

def color_for(verdict: str) -> str:
    return {"phishing":"#ff5d73","ai_phish":"#ff5d73",
            "suspicious":"#f5c451","clean":"#35d07f"}.get(verdict,"#4da3ff")

def human_verdict(verdict: str):
    if verdict == "clean":
        return "EMAIL APPEARS SAFE","No significant phishing indicators were detected.","clean","✅"
    if verdict == "ai_phish":
        return "AI-GENERATED PHISHING DETECTED","The email shows strong indicators of AI-assisted phishing.","phishing","🚨"
    if verdict == "phishing":
        return "PHISHING DETECTED","This email contains indicators commonly associated with phishing.","phishing","🚨"
    return "SUSPICIOUS EMAIL","This email requires additional attention.","suspicious","⚠️"

if analyze and raw_email.strip():
    try:
        layer1_mod, predict_mod, pipeline_mod, classifier, layer3_mod = load_system(model_path)
    except Exception as e:
        st.error(f"Unable to start email analysis: {e}")
        st.stop()

    result = pipeline_mod.run_pipeline(
        raw_email.encode(), layer1_mod, classifier, predict_mod.load_eml_text,
        always_run_layer2=ensemble, layer3_mod=None
    )

    verdict = result.get("final_verdict","suspicious")
    score = float(result.get("final_risk_score") or 0)
    title, description, css_class, icon = human_verdict(verdict)
    accent = color_for(verdict)

    st.markdown(f"""
    <div class="result-card">
      <div class="verdict {css_class}">
        <div class="verdict-icon">{icon}</div>
        <div class="verdict-title">{title}</div>
        <div class="verdict-desc">{description}</div>
        <div class="risk">{score:.0f}<span style="font-size:1rem;color:#9aa8bf"> / 100 risk</span></div>
        <div class="risk-track"><div class="risk-fill" style="width:{max(0,min(100,score))}%;background:{accent}"></div></div>
      </div>
    """, unsafe_allow_html=True)

    l1 = result.get("layer1") or {}
    l2 = result.get("layer2") or {}
    auth = l1.get("auth") or {}

    def status(value):
        value = str(value or "").lower()
        if value == "pass": return '<span class="good">✓ Verified</span>'
        if value == "fail": return '<span class="bad">✕ Failed</span>'
        return '<span class="neutral">— Not available</span>'

    def risk_status():
        if score >= 70: return '<span class="bad">High risk</span>'
        if score >= 30: return '<span class="warn">Review</span>'
        return '<span class="good">Low risk</span>'

    a,b = st.columns(2)
    with a:
        st.markdown(f"""
        <div class="info-card">
          <div class="info-title">Sender Verification</div>
          <div class="info-row"><span>SPF</span>{status(auth.get("spf"))}</div>
          <div class="info-row"><span>DKIM</span>{status(auth.get("dkim"))}</div>
          <div class="info-row"><span>DMARC</span>{status(auth.get("dmarc"))}</div>
          <div class="info-row"><span>Security risk</span>{risk_status()}</div>
        </div>
        """, unsafe_allow_html=True)

    with b:
        prediction = l2.get("predicted_label")
        pred_display = {"ham":"Likely legitimate","phishing":"Likely phishing",
                        "ai_phish":"AI-generated phishing"}.get(prediction,"Not required")
        confidence = float(l2.get("confidence") or 0)*100
        st.markdown(f"""
        <div class="info-card">
          <div class="info-title">Email Content Analysis</div>
          <div class="info-row"><span>Assessment</span><span class="neutral">{pred_display}</span></div>
          <div class="info-row"><span>Confidence</span><span class="neutral">{confidence:.1f}%</span></div>
          <div class="info-row"><span>Analysis status</span><span class="good">✓ Complete</span></div>
        </div>
        """, unsafe_allow_html=True)

    reasons = l1.get("reasons") or []
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="info-card"><div class="info-title">Why this result?</div>', unsafe_allow_html=True)

    if reasons:
        for reason in reasons:
            st.markdown(f'<div class="reason">• {reason}</div>', unsafe_allow_html=True)
    elif verdict == "clean":
        st.markdown('<div class="reason">• No significant security rules were triggered.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="reason">• The email content contains patterns requiring attention.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if l2:
        with st.expander("View analysis confidence"):
            labels = {"ham":"Legitimate","phishing":"Phishing","ai_phish":"AI-generated phishing"}
            for lbl,p in (l2.get("probabilities") or {}).items():
                value = float(p)*100
                st.markdown(f"""
                <div class="confidence-row">
                  <div class="confidence-head"><span>{labels.get(lbl,lbl)}</span><span>{value:.1f}%</span></div>
                  <div class="confidence-track"><div class="confidence-fill" style="width:{value}%"></div></div>
                </div>
                """, unsafe_allow_html=True)

    with st.expander("Technical details"):
        st.json(result)

    st.markdown('</div>', unsafe_allow_html=True)

elif analyze:
    st.warning("Please paste an email or upload a .eml file first.")

st.markdown(
    '<div class="footer-note">PhishShield AI · Automated email security analysis · '
    'Use results as a security aid, not as the sole basis for high-impact decisions.</div>',
    unsafe_allow_html=True
)
