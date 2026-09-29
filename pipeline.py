"""
PhishShield AI — Pipeline Orchestrator

Flow:
    EMAIL -> Layer 1 (rules/headers) -> Layer 2 (DistilBERT) -> fusion -> Layer 3 -> FINAL

  * Layer 1 runs on EVERY email (free; catches header/domain attacks).
  * Layer 2 runs unless Layer 1 is confident the email is clean AND authenticated.
  * fuse() combines Layers 1+2 into a numeric assessment. It is EVIDENCE for
    Layer 3, and it is kept in the output as `fusion` (needed for ablation).
  * Layer 3 (GPT adjudicator) runs on EVERY email and is the final authority.
    Its verdict is copied into `final` and is never modified afterwards.
  * If Layer 3 cannot produce a verdict (no key, API error, invalid reply) the
    result falls back to the fused assessment and is marked `degraded: true`,
    `decided_by: "fusion_fallback"`. It is never presented as a GPT decision.

Output keys
  final            Layer 3's decision (new vocabulary: likely_legitimate,
                   needs_verification, phishing_threat, ai_assisted_phishing)
  final_verdict    COMPATIBILITY: the same decision in the old vocabulary
  final_risk_score   (clean / suspicious / phishing / ai_phish) so the existing
                   React app and Streamlit dashboard keep working unchanged.
                   Derived mechanically from `final`; not an independent score.
  fusion           the pre-Layer-3 fused assessment
  layer1, layer2, layer3   each layer's evidence

Usage:
    python pipeline.py --model ./Layer-2/models/phishing-model-3class --eml ./Layer-1/sample_phish.eml --pretty
    python pipeline.py --model ./Layer-2/models/phishing-model-3class --dir ./some_emails --pretty
    python pipeline.py --model ... --eml x.eml --no-layer3     # Layers 1+2+fusion only (ablation / offline)
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

    IMPORTANT — verified-clean vs. no-data:
    Layer 1 scoring 0 means two very different things depending on WHY:
      (a) it actively verified SPF/DKIM/DMARC pass and found no lookalike/
          urgency/mismatch signals — a real, earned "clean" reading, or
      (b) the input had no parseable headers at all (common when someone
          pastes plain text rather than a raw .eml) — Layer 1 has NO
          evidence either way, positive or negative.
    Treating (b) the same as (a) let a single, uncorroborated, possibly
    miscalibrated Layer 2 reading alone push the score into "phishing"
    territory with no grounding at all. We now require a bit more from
    Layer 2 before escalating hard when Layer 1 has nothing to corroborate
    with — this doesn't touch the case that matters most for the hybrid
    design (a VERIFIED-clean sender with phishy language), which still
    escalates exactly as aggressively as before.
    """
    l1_score = layer1_result.get("infra_risk_score", 0.0)
    auth = layer1_result.get("auth", {}) or {}
    reasons = layer1_result.get("reasons", []) or []

    # Layer 1 "had data" if it actually observed auth results one way or
    # another, or found any rule-based signal worth reporting. If every
    # auth field is null AND there are no reasons, Layer 1 saw nothing to
    # judge — that's an abstention, not a clean bill of health.
    l1_had_data = any(auth.get(k) is not None for k in ("spf", "dkim", "dmarc")) or bool(reasons)

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

        # Weighted blend, so both layers' evidence is represented
        blended = 0.5 * l1_score + 0.5 * l2_score

        if l1_had_data:
            # Layer 1 actually verified something (pass, fail, or a rule
            # fired) — trust a confident Layer 2 reading fully, since this
            # is exactly the "verified-clean sender, phishy language" case
            # the hybrid architecture exists to catch.
            final = max(blended, l1_score, l2_score * 0.9)
        else:
            # Layer 1 has no corroborating data at all (e.g. pasted plain
            # text with no real headers). A single uncorroborated Layer 2
            # reading still counts, but needs to be more confident to reach
            # the same escalation, and it can't outrun the blended average
            # on its own the way a corroborated reading can.
            final = max(blended, l2_score * 0.75)

    final = round(min(final, 100.0), 1)

    # Base verdict from the fused score (recall-favoring thresholds)
    if final >= 60:
        verdict = "phishing"
    elif final >= 30:
        verdict = "suspicious"
    else:
        verdict = "clean"

    # Without ANY corroborating infrastructure evidence — no headers to check
    # at all, not "verified clean", just nothing to examine — a single text
    # classifier's opinion, however confident, shouldn't alone justify the
    # system declaring definitive PHISHING. It's still flagged for review
    # (nothing is silently cleared), just not asserted with full confidence
    # on an uncorroborated signal. This rule is general and content-blind:
    # it applies identically to every header-less input, so it can't be
    # mistaken for a carve-out for any particular email.
    if not l1_had_data and verdict == "phishing":
        verdict = "suspicious"

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
        "layer1_had_data": l1_had_data,
    }


# Layer 3's vocabulary <-> the older fused vocabulary used by the current frontend.
_L3_TO_LEGACY = {
    "likely_legitimate": "clean",
    "needs_verification": "suspicious",
    "phishing_threat": "phishing",
    "ai_assisted_phishing": "ai_phish",
}
_FUSION_TO_L3 = {v: k for k, v in _L3_TO_LEGACY.items()}


def build_final(l3: dict | None, fused: dict, l1_summary: dict) -> dict:
    """Assemble the `final` block.

    * Layer 3 produced a verdict  -> copy it verbatim. Nothing modifies it.
    * Layer 3 was skipped (--no-layer3) -> fused assessment, decided_by "fusion_only".
    * Layer 3 failed              -> fused assessment, marked degraded, with the reason.
    """
    if l3 is not None and l3.get("status") == "ok":
        return {
            "verdict": l3["final_verdict"],
            "risk_score": l3["risk_score"],
            "confidence": l3["confidence"],
            "classification": l3["classification"],
            "debrief": l3["debrief"],
            "evidence": l3["evidence"],
            "recommended_action": l3["recommended_action"],
            "layer2_supported": l3["layer2_supported"],
            "guardrails_applied": l3.get("guardrails_applied", []),
            "decided_by": "layer3",
            "degraded": False,
        }

    verdict = _FUSION_TO_L3.get(fused["final_verdict"], "needs_verification")
    if l3 is None:
        why, decided_by, degraded = "Layer 3 was not run for this request.", "fusion_only", False
    else:
        why = (f"Layer 3 could not produce a verdict ({l3.get('status')}: {l3.get('detail')}). "
               f"This is the Layer 1 + Layer 2 fused assessment, not a Layer 3 decision.")
        decided_by, degraded = "fusion_fallback", True
    return {
        "verdict": verdict,
        "risk_score": fused["final_risk_score"],
        "confidence": None,
        "classification": None,
        "debrief": why,
        "evidence": list(l1_summary.get("reasons") or []),
        "recommended_action": ("Treat as provisional and re-run when Layer 3 is available."
                               if degraded else "Review the Layer 1 and Layer 2 evidence."),
        "layer2_supported": None,
        "guardrails_applied": [],
        "decided_by": decided_by,
        "degraded": degraded,
    }


def run_pipeline(raw_eml: bytes, layer1_mod, classifier, load_eml_text_fn,
                 always_run_layer2: bool = False, layer3_mod=None,
                 l3_model: str | None = None) -> dict:
    # ---- Layer 1 ----
    l1 = layer1_mod.analyze_email(raw_eml).to_dict()

    # ---- Decide whether to run Layer 2 (unchanged) ----
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

    # ---- Layer 3: runs on EVERY email and is the final authority ----
    l3 = None
    if layer3_mod is not None:
        l3 = layer3_mod.adjudicate(raw_eml, l1, l2, fused, model=l3_model)

    final = build_final(l3, fused, l1_summary)
    layer3_ran = bool(l3 and l3.get("status") == "ok")

    # Nothing below this line changes `final`. The top-level final_verdict /
    # final_risk_score are a mechanical translation of it for the existing UI.
    return {
        "final": final,
        "final_verdict": _L3_TO_LEGACY[final["verdict"]],
        "final_risk_score": final["risk_score"],
        "layer2_phish_probability": fused["layer2_phish_probability"],
        "layer2_predicted_label": fused["layer2_predicted_label"],
        "layer1_had_data": fused["layer1_had_data"],
        "layer2_ran": layer2_ran,
        "layer3_ran": layer3_ran,
        "fusion": fused,
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
                    help="Accepted for backward compatibility. Layer 3 now runs by default.")
    ap.add_argument("--no-layer3", action="store_true",
                    help="Skip Layer 3 (Layers 1+2+fusion only). For ablation runs / offline use.")
    ap.add_argument("--l3-model", default=None,
                    help="Override the Layer 3 model (default: PHISHSHIELD_L3_MODEL or gpt-6-sol)")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    layer1_mod = _load_module("layer1_detector", root / "Layer-1" / "layer1_detector.py")
    predict_mod = _load_module("predict", root / "Layer-2" / "predict.py")

    layer3_mod = None
    if not args.no_layer3:
        layer3_mod = _load_module("layer3_attribution", root / "Layer-3" / "layer3_attribution.py")

    classifier = predict_mod.PhishClassifier(args.model)
    indent = 2 if args.pretty else None

    if args.dir:
        results = []
        for path in sorted(Path(args.dir).rglob("*.eml")):
            r = run_pipeline(path.read_bytes(), layer1_mod, classifier,
                             predict_mod.load_eml_text, args.always_run_layer2,
                             layer3_mod=layer3_mod, l3_model=args.l3_model)
            r["file"] = path.name
            results.append(r)
        print(json.dumps(results, indent=indent, default=str))
        return

    if not args.eml:
        ap.error("Provide --eml or --dir")

    raw = Path(args.eml).read_bytes()
    result = run_pipeline(raw, layer1_mod, classifier,
                          predict_mod.load_eml_text, args.always_run_layer2,
                          layer3_mod=layer3_mod, l3_model=args.l3_model)
    print(json.dumps(result, indent=indent, default=str))


if __name__ == "__main__":
    main()