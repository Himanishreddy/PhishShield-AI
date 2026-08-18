"""
PhishShield AI — Pipeline Orchestrator

Ties Layer 1 (deterministic rules) and Layer 2 (DistilBERT) into one
hybrid detector, emitting a single combined verdict per email.

Design (the "hybrid" strategy discussed):
  * Layer 1 runs on EVERY email — it's free and catches header/domain
    attacks the text model can't see (spoofing, DMARC fail, lookalikes).
  * Layer 2 runs UNLESS Layer 1 is highly confident the email is clean
    AND came from an authenticated sender. This saves compute while
    still defending against text-only phishing that Layer 1 can't catch.
  * The two signals are fused into one final_risk_score (0-100) and a
    final_verdict, with both layers' evidence preserved for the analyst.

This orchestrator imports the two existing modules rather than
reimplementing them, so there's one source of truth for each layer.

Expected layout (adjust --model / import paths to match yours):
    Phishing/
      Layer-1/layer1_detector.py
      Layer-2/predict.py
      Layer-2/models/phishing-model/

Usage:
    python pipeline.py --model ./Layer-2/models/phishing-model --eml ./Layer-1/sample_phish.eml --pretty
    python pipeline.py --model ./Layer-2/models/phishing-model --dir ./some_emails --pretty
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Import the two layer modules by file path (robust to folder layout)
# ---------------------------------------------------------------------------

def _load_module(name: str, path: Path):
    if not path.exists():
        sys.exit(f"Could not find {path}. Adjust the path in pipeline.py or pass the right --root.")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so @dataclass can resolve type hints
    # via cls.__module__ (required on Python 3.12+ / 3.14).
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fusion logic
# ---------------------------------------------------------------------------

def fuse(layer1_result: dict, layer2_result: dict | None) -> dict:
    """Combine Layer 1 + Layer 2 into a single verdict.

    Weighting rationale:
      - Layer 1 covers infrastructure (spoofing, auth). A hard auth failure
        or clear spoof is strong evidence on its own.
      - Layer 2 covers language/intent. High phishing probability is strong
        evidence even when the headers look clean.
    We take a weighted blend but let either layer escalate on its own, since
    a phish only needs ONE layer to catch it (favoring recall).
    """
    l1_score = layer1_result.get("infra_risk_score", 0.0)

    if layer2_result is None:
        # Layer 2 was skipped (Layer 1 confident-clean). Verdict rests on L1.
        final = l1_score
        l2_phish_prob = None
    else:
        probs = layer2_result.get("probabilities", {})
        # phishing prob = 1 - ham prob (works for binary and multiclass)
        ham_prob = probs.get("ham", 0.0)
        l2_phish_prob = round(1.0 - ham_prob, 4)
        l2_score = l2_phish_prob * 100

        # Weighted blend, but take the max so either layer can escalate alone
        blended = 0.5 * l1_score + 0.5 * l2_score
        final = max(blended, l1_score, l2_score * 0.9)

    final = round(min(final, 100.0), 1)

    # Base verdict from the fused score (recall-favoring thresholds)
    if final >= 60:
        verdict = "phishing"
    elif final >= 30:
        verdict = "suspicious"
    else:
        verdict = "clean"

    # Preserve the AI-vs-human phishing distinction — the project's core claim.
    # If the email is judged phishing AND Layer 2 specifically identified it as
    # AI-generated, surface that as the final verdict instead of the generic
    # "phishing". We only do this when the model actually predicted ai_phish
    # (not merely when that probability is nonzero), so it stays trustworthy.
    l2_label = None
    if layer2_result is not None:
        l2_label = layer2_result.get("predicted_label")
    if verdict == "phishing" and l2_label == "ai_phish":
        verdict = "ai_phish"

    return {
        "final_verdict": verdict,
        "final_risk_score": final,
        "layer2_phish_probability": l2_phish_prob,
        "layer2_predicted_label": l2_label,
    }


def run_pipeline(raw_eml: bytes, layer1_mod, classifier, load_eml_text_fn,
                 always_run_layer2: bool = False, layer3_mod=None) -> dict:
    # ---- Layer 1 ----
    l1 = layer1_mod.analyze_email(raw_eml).to_dict()

    # ---- Decide whether to run Layer 2 ----
    l1_score = l1.get("infra_risk_score", 0.0)
    auth = l1.get("auth", {})
    authenticated = (auth.get("spf") == "pass" and auth.get("dkim") == "pass"
                     and auth.get("dmarc") == "pass")
    confident_clean = l1_score < 15 and authenticated

    from email import message_from_bytes
    msg = message_from_bytes(raw_eml)

    l2 = None
    layer2_ran = False
    if always_run_layer2 or not confident_clean:
        # Extract the same text representation the model was trained on
        text = _extract_text(msg)
        if text.strip():
            l2 = classifier.predict(text)
            layer2_ran = True

    fused = fuse(l1, l2)

    l1_summary = {
        "infra_risk_score": l1.get("infra_risk_score"),
        "verdict": l1.get("verdict"),
        "auth": l1.get("auth"),
        "reasons": l1.get("reasons"),
        "from_address": l1.get("from_address"),
        "subject": l1.get("subject"),
    }

    # ---- Layer 3: attribution — ONLY on confirmed phishing ----
    # Matches the architecture: expensive LLM analysis runs on the few
    # confirmed threats, not on every email.
    l3 = None
    layer3_ran = False
    if layer3_mod is not None and fused["final_verdict"] == "phishing":
        subject = l1.get("subject", "") or msg.get("Subject", "") or ""
        body = _extract_text(msg)
        l3 = layer3_mod.attribute(subject, body, layer1_evidence=l1_summary)
        layer3_ran = l3.get("_meta", {}).get("status") == "ok"

    return {
        **fused,
        "layer2_ran": layer2_ran,
        "layer3_ran": layer3_ran,
        "layer1": l1_summary,
        "layer2": l2,
        "layer3": l3,
    }


def _extract_text(msg) -> str:
    """Subject + body, matching how the model was trained."""
    import re
    subject = msg.get("Subject", "") or ""

    def strip_html(html: str) -> str:
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()

    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    parts.append(payload.decode(errors="ignore"))
            elif part.get_content_type() == "text/html" and not parts:
                payload = part.get_payload(decode=True)
                if payload:
                    parts.append(strip_html(payload.decode(errors="ignore")))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            text = payload.decode(errors="ignore")
            if msg.get_content_type() == "text/html":
                text = strip_html(text)
            parts.append(text)

    body = "\n".join(parts).strip()
    return f"{subject}\n\n{body}".strip() if subject else body


def main():
    ap = argparse.ArgumentParser(description="PhishShield end-to-end pipeline")
    ap.add_argument("--model", required=True, help="Path to the Layer 2 model folder")
    ap.add_argument("--eml", help="Single .eml file")
    ap.add_argument("--dir", help="Folder of .eml files")
    ap.add_argument("--root", default=".", help="Project root (where Layer-1/ and Layer-2/ live)")
    ap.add_argument("--always-run-layer2", action="store_true",
                    help="Run Layer 2 on every email (ensemble mode) instead of gating")
    ap.add_argument("--layer3", action="store_true",
                    help="Run Layer 3 LLM attribution on confirmed phishing (needs Ollama running)")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    layer1_mod = _load_module("layer1_detector", root / "Layer-1" / "layer1_detector.py")
    predict_mod = _load_module("predict", root / "Layer-2" / "predict.py")

    layer3_mod = None
    if args.layer3:
        layer3_mod = _load_module("layer3_attribution", root / "Layer-3" / "layer3_attribution.py")

    classifier = predict_mod.PhishClassifier(args.model)
    indent = 2 if args.pretty else None

    if args.dir:
        results = []
        for path in sorted(Path(args.dir).rglob("*.eml")):
            r = run_pipeline(path.read_bytes(), layer1_mod, classifier,
                             predict_mod.load_eml_text, args.always_run_layer2,
                             layer3_mod=layer3_mod)
            r["file"] = path.name
            results.append(r)
        print(json.dumps(results, indent=indent, default=str))
        return

    if not args.eml:
        ap.error("Provide --eml or --dir")

    raw = Path(args.eml).read_bytes()
    result = run_pipeline(raw, layer1_mod, classifier,
                          predict_mod.load_eml_text, args.always_run_layer2,
                          layer3_mod=layer3_mod)
    print(json.dumps(result, indent=indent, default=str))


if __name__ == "__main__":
    main()