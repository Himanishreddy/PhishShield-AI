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
# Sidebar — configuration
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Configuration")

# Auto-discover trained models under Layer-2/models
models_dir = ROOT / "Layer-2" / "models"
available = []
if models_dir.exists():
    available = [str(p) for p in models_dir.iterdir() if p.is_dir()]

if available:
    model_path = st.sidebar.selectbox("Detection model", available,
                                      index=len(available) - 1,
                                      help="3-class model reports ham / phishing / ai_phish")
else:
    model_path = st.sidebar.text_input(
        "Model path", value=str(ROOT / "Layer-2" / "models" / "phishing-model"))

ensemble = st.sidebar.checkbox(
    "Ensemble mode (run Layer 2 on every email)", value=False,
    help="Off = gate: Layer 2 is skipped when Layer 1 is confident-clean and the "
         "sender is authenticated. On = Layer 2 always runs.")

run_layer3 = st.sidebar.checkbox(
    "Layer 3 attribution (confirmed phishing only)", value=False,
    help="Runs a local LLM (Ollama) to infer attacker methodology on emails "
         "judged phishing. Needs Ollama running with a model pulled.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span class='tag' style='color:#8b98a5;font-family:IBM Plex Mono,monospace;"
    "font-size:0.72rem'>Layer 1 · rules & headers<br>Layer 2 · DistilBERT semantic<br>"
    "Fusion · recall-favoring</span>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------

st.markdown("""
<div class="ps-masthead">
  <h1>PhishShield<span style="color:#39c5cf">/</span>SOC</h1>
  <span class="tag">Security Operations · Triage Console</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

SAMPLE = """From: "Microsoft Support" <security-update@micros0ft-support.com>
Reply-To: attacker-collect@totally-diff-domain.ru
To: cfo@yourcompany.com
Subject: URGENT: Action Required Immediately - Account Suspended
Authentication-Results: mx.company.com; spf=fail; dkim=fail; dmarc=fail
Content-Type: text/html

Your account will be locked within 2 hours due to unauthorized login attempt.
Click here to verify your identity: http://secure-login-portal.xyz/verify
"""

col_in, col_btn = st.columns([5, 1])
with col_in:
    raw_email = st.text_area("Raw email (paste full .eml including headers)",
                             value=SAMPLE, height=200, label_visibility="collapsed")
with col_btn:
    analyze = st.button("Analyze", type="primary", use_container_width=True)
    if st.button("Clear", use_container_width=True):
        st.rerun()

uploaded = st.file_uploader("…or upload a .eml file", type=["eml"], label_visibility="collapsed")
if uploaded is not None:
    raw_email = uploaded.read().decode(errors="ignore")


# ---------------------------------------------------------------------------
# Analysis + render
# ---------------------------------------------------------------------------

def color_for(verdict: str) -> str:
    return {"phishing": "#f0506e", "ai_phish": "#f0506e", "suspicious": "#d9a441",
            "clean": "#3fb950"}.get(verdict, "#8b98a5")


if analyze and raw_email.strip():
    try:
        layer1_mod, predict_mod, pipeline_mod, classifier, layer3_mod = load_system(model_path)
    except Exception as e:
        st.error(f"Couldn't load the detection system: {e}")
        st.stop()

    raw_bytes = raw_email.encode()
    result = pipeline_mod.run_pipeline(
        raw_bytes, layer1_mod, classifier, predict_mod.load_eml_text,
        always_run_layer2=ensemble,
        layer3_mod=(layer3_mod if run_layer3 else None))

    verdict = result["final_verdict"]
    score = result["final_risk_score"]
    accent = color_for(verdict)
    # ai_phish shares the phishing (red) styling and gets a readable label
    css_class = "phishing" if verdict == "ai_phish" else verdict
    verdict_label = "AI-GENERATED PHISHING" if verdict == "ai_phish" else verdict.upper()

    # ---- Verdict banner (signature element) ----
    st.markdown(f"""
    <div class="verdict {css_class}">
      <div class="label">Final verdict · fused risk score</div>
      <div class="value">{verdict_label} · {score:.0f}/100</div>
      <div class="score-track">
        <div class="score-fill" style="width:{score}%;background:{accent}"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    l1 = result["layer1"]
    l2 = result["layer2"]

    c1, c2 = st.columns(2)

    # ---- Layer 1 panel ----
    with c1:
        auth = l1.get("auth", {})
        def auth_row(k, v):
            cls = "pass" if v == "pass" else ("fail" if v == "fail" else "")
            return f"<div class='kv'><span class='k'>{k}</span><span class='v {cls}'>{v or '—'}</span></div>"

        reasons_html = "".join(f"<div class='reason'>{r}</div>"
                               for r in l1.get("reasons", [])) or \
                       "<div class='kv'><span class='v'>No rule-based flags</span></div>"

        st.markdown(f"""
        <div class="panel">
          <h3>Layer 1 — Metadata & Rules</h3>
          <div class="kv"><span class="k">infra_risk</span><span class="v">{l1.get('infra_risk_score')}/100</span></div>
          {auth_row('spf', auth.get('spf'))}
          {auth_row('dkim', auth.get('dkim'))}
          {auth_row('dmarc', auth.get('dmarc'))}
          <div class="kv"><span class="k">from</span><span class="v">{(l1.get('from_address') or '—')[:34]}</span></div>
          <div style="margin-top:0.9rem">{reasons_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Layer 2 panel ----
    with c2:
        if l2 is None:
            st.markdown(f"""
            <div class="panel">
              <h3>Layer 2 — DistilBERT Semantic</h3>
              <div class="gate-note">Layer 2 was skipped. Layer 1 judged this email
              confidently clean from an authenticated sender, so the semantic model
              wasn't needed. Enable Ensemble mode to force it.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            probs = l2.get("probabilities", {})
            prob_html = ""
            for lbl, p in probs.items():
                prob_html += f"""
                <div class="prob-row">
                  <div class="plabel"><span>{lbl}</span><span>{p*100:.1f}%</span></div>
                  <div class="prob-track"><div class="prob-fill" style="width:{p*100}%"></div></div>
                </div>"""
            st.markdown(f"""
            <div class="panel">
              <h3>Layer 2 — DistilBERT Semantic</h3>
              <div class="kv"><span class="k">predicted</span><span class="v">{l2.get('predicted_label')}</span></div>
              <div class="kv"><span class="k">confidence</span><span class="v">{l2.get('confidence',0)*100:.1f}%</span></div>
              <div style="margin-top:0.9rem">{prob_html}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Analyst note on how the verdict was reached ----
    ran = "both layers ran" if result.get("layer2_ran") else "Layer 1 only (Layer 2 gated out)"
    st.markdown(f"""
    <div class="gate-note" style="margin-top:1.2rem">
      DECISION PATH · {ran}. Either layer can raise an alert on its own — the fusion
      favors recall, because a missed phish costs more than a second look at a clean email.
    </div>
    """, unsafe_allow_html=True)

    # ---- Layer 3 attribution panel ----
    l3 = result.get("layer3")
    if l3 is not None:
        meta = l3.get("_meta", {})
        if meta.get("status") == "ok":
            triggers = l3.get("psychological_triggers", [])
            trig_html = "".join(
                f"<span style='display:inline-block;background:#1c232d;border:1px solid #2a333f;"
                f"border-radius:99px;padding:0.15rem 0.7rem;margin:0.15rem 0.25rem 0.15rem 0;"
                f"font-family:IBM Plex Mono,monospace;font-size:0.76rem;color:#d9a441'>{t}</span>"
                for t in triggers)
            indicators = l3.get("key_indicators", [])
            ind_html = "".join(f"<div class='reason'>{i}</div>" for i in indicators)

            st.markdown(f"""
            <div class="panel" style="margin-top:1.2rem;border-left:4px solid #39c5cf">
              <h3>Layer 3 — Threat Attribution (inferred)</h3>
              <div class="kv"><span class="k">primary_objective</span><span class="v">{l3.get('primary_objective','—')}</span></div>
              <div class="kv"><span class="k">target_persona</span><span class="v">{l3.get('target_persona','—')}</span></div>
              <div class="kv"><span class="k">sophistication</span><span class="v">{l3.get('sophistication','—')}</span></div>
              <div style="margin:0.7rem 0 0.3rem 0"><span class="k" style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;color:#8b98a5">psychological_triggers</span></div>
              <div style="margin-bottom:0.6rem">{trig_html}</div>
              <div style="margin:0.7rem 0 0.3rem 0"><span class="k" style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;color:#8b98a5">key_indicators</span></div>
              {ind_html}
              <div style="margin-top:0.9rem;padding:0.7rem 0.8rem;background:#12171d;border-radius:6px;border:1px solid #2a333f">
                <div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#39c5cf;letter-spacing:0.1em;margin-bottom:0.4rem">ANALYST SUMMARY</div>
                <div style="font-size:0.9rem;color:#e6edf3">{l3.get('analyst_summary','—')}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Illustrative generation prompt (hypothesis — not a recovered attacker input)"):
                st.markdown(
                    "<div class='gate-note' style='margin-bottom:0.6rem'>This is an example of the "
                    "<b>kind</b> of instruction that could produce a similar email, generated for "
                    "defensive understanding. It is NOT the attacker's real prompt — an LLM cannot "
                    "recover that.</div>", unsafe_allow_html=True)
                st.code(l3.get("illustrative_generation_prompt", "—"), language="text")

            st.caption(f"⚠ {meta.get('disclaimer','')}")
        else:
            st.markdown(f"""
            <div class="panel" style="margin-top:1.2rem;border-left:4px solid #8b98a5">
              <h3>Layer 3 — Threat Attribution</h3>
              <div class="gate-note">{meta.get('detail', 'Layer 3 did not run.')}</div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("Raw verdict JSON (for report / debugging)"):
        st.json(result)

elif analyze:
    st.warning("Paste an email first — headers included, so Layer 1 can read the authentication results.")
else:
    st.markdown(
        "<div class='gate-note'>Paste an email above and hit Analyze. "
        "The sample loaded is a spoofed-Microsoft credential-harvest — try it, "
        "then swap in a clean email to watch the gate skip Layer 2.</div>",
        unsafe_allow_html=True)