"""
PhishShield AI — Layer 3: Threat Attribution (LLM reverse-engineering)

Runs ONLY on emails the pipeline has already flagged as phishing. Passes the
email + Layer 1/2 evidence to a local LLM (via Ollama) and asks it to infer
the attacker's methodology, producing structured threat intelligence for the
SOC analyst.

IMPORTANT — HONEST FRAMING (read this, and put it in your report):
    This layer does NOT recover the attacker's real prompt or identity. An LLM
    cannot forensically reconstruct the exact input another model was given.
    What it produces is *inferred* threat intelligence — a plausible analysis
    of objective, psychological triggers, and target persona, plus an
    ILLUSTRATIVE example of the kind of prompt that could generate such an
    email. Treat every field as an analyst's educated hypothesis, not fact.
    The value is triage acceleration and pattern-spotting, not attribution
    certainty.

Requires a running Ollama instance:
    1. Install from https://ollama.com
    2. ollama pull llama3.2
    3. Ollama serves on http://localhost:11434 automatically.

This module talks to Ollama over its local HTTP API using only the standard
library (urllib) — no extra pip packages needed.

Usage (standalone, for testing):
    python layer3_attribution.py --eml ./Layer-1/sample_phish.eml

Usually it's called by the pipeline/dashboard, not directly.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
import urllib.error
from pathlib import Path


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"


# The instruction to the analysis model. We force JSON output and label every
# speculative field as inferred, so the model doesn't present guesses as fact.
SYSTEM_INSTRUCTION = """You are a threat-intelligence analyst assisting a Security Operations Center. \
You are given an email that has ALREADY been confirmed as phishing by an upstream detector. \
Your job is to infer the attacker's likely methodology to help the SOC team understand and \
defend against this campaign.

You MUST return ONLY a valid JSON object (no prose, no markdown fences) with these keys:
  "primary_objective": one of ["Credential Harvesting", "Malware Delivery", "Wire Transfer Fraud", "Data Exfiltration", "Reconnaissance", "Other"]
  "psychological_triggers": array of strings from ["Authority", "Urgency", "Fear", "Curiosity", "Greed", "Trust", "Scarcity", "Social Proof"]
  "target_persona": short string describing the role this email seems aimed at (e.g. "Finance executive", "IT administrator", "General employee")
  "sophistication": one of ["Low", "Medium", "High"]
  "key_indicators": array of short strings — the specific textual/structural cues that informed your analysis
  "illustrative_generation_prompt": a SINGLE string. This is an ILLUSTRATIVE EXAMPLE of the kind of instruction an attacker might have given a text generator to produce a similar email. It is a hypothesis for defensive understanding, NOT a reconstruction of any real prompt.
  "analyst_summary": 1-2 sentences a SOC analyst can read at a glance.

Base every field only on what is present in the email. Do not invent specific names, IPs, or URLs that aren't there."""


def load_eml_text(path: Path) -> tuple[str, str]:
    """Return (subject, body) from a .eml. Mirrors the other layers' parsing."""
    from email import message_from_bytes

    msg = message_from_bytes(path.read_bytes())
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

    return subject.strip(), "\n".join(parts).strip()


def ollama_available(model: str = DEFAULT_MODEL, timeout: float = 2.0) -> tuple[bool, str]:
    """Quick health check: is Ollama running and does it have the model?"""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        models = [m.get("name", "") for m in data.get("models", [])]
        has_model = any(model in m for m in models)
        if not has_model:
            return False, (f"Ollama is running but '{model}' isn't pulled. "
                           f"Run: ollama pull {model}. Available: {models or 'none'}")
        return True, "ok"
    except urllib.error.URLError:
        return False, ("Ollama isn't reachable at localhost:11434. "
                       "Install from ollama.com and make sure it's running.")
    except Exception as e:
        return False, f"Ollama check failed: {e}"


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of the model's response, tolerating stray text."""
    # Fast path
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("Model did not return parseable JSON")


def attribute(subject: str, body: str, layer1_evidence: dict | None = None,
              model: str = DEFAULT_MODEL, timeout: float = 120.0) -> dict:
    """Run the attribution analysis. Returns a structured dict (see keys above),
    always including a `_meta` block with the honesty disclaimer and status."""

    disclaimer = ("Inferred threat intelligence — hypotheses for defensive triage, "
                  "not forensic attribution. The generation prompt is illustrative, "
                  "not a recovered attacker input.")

    ok, msg = ollama_available(model)
    if not ok:
        return {
            "_meta": {"status": "unavailable", "detail": msg, "disclaimer": disclaimer},
        }

    evidence_str = ""
    if layer1_evidence:
        auth = layer1_evidence.get("auth", {})
        evidence_str = (f"\n\nUpstream detector evidence (for context):\n"
                        f"- Authentication: spf={auth.get('spf')}, dkim={auth.get('dkim')}, dmarc={auth.get('dmarc')}\n"
                        f"- Sender: {layer1_evidence.get('from_address')}\n"
                        f"- Flags: {layer1_evidence.get('reasons')}")

    prompt = (f"{SYSTEM_INSTRUCTION}\n\n"
              f"=== EMAIL UNDER ANALYSIS ===\n"
              f"Subject: {subject}\n\n{body}{evidence_str}\n\n"
              f"Return the JSON object now:")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",          # ask Ollama to constrain output to JSON
        "options": {"temperature": 0.3},
    }

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        raw_text = result.get("response", "")
        analysis = _extract_json(raw_text)
        analysis["_meta"] = {"status": "ok", "model": model, "disclaimer": disclaimer}
        return analysis
    except urllib.error.URLError as e:
        return {"_meta": {"status": "error", "detail": f"Ollama request failed: {e}",
                          "disclaimer": disclaimer}}
    except ValueError as e:
        return {"_meta": {"status": "parse_error", "detail": str(e),
                          "raw": raw_text[:500], "disclaimer": disclaimer}}


def main():
    ap = argparse.ArgumentParser(description="PhishShield Layer 3 — LLM attribution")
    ap.add_argument("--eml", help="Path to a .eml file to analyze")
    ap.add_argument("--text", help="Raw email text to analyze")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    if args.eml:
        subject, body = load_eml_text(Path(args.eml))
    elif args.text:
        subject, body = "", args.text
    else:
        ap.error("Provide --eml or --text")

    result = attribute(subject, body, model=args.model)
    print(json.dumps(result, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()