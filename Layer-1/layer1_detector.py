"""
PhishShield AI — Layer 1: Deterministic Metadata Filter

Parses a raw .eml message and produces a structured risk assessment based on
authentication results, sender/domain anomalies, and header inconsistencies.
No ML involved here — this is the fast, cheap triage gate that decides
whether a message needs to go to Layer 2 (semantic classifier).

Usage:
    python layer1_detector.py path/to/message.eml

Or import and call `analyze_email(raw_bytes_or_str)` directly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from email import message_from_string, message_from_bytes
from email.message import Message
from email.utils import parseaddr, parsedate_to_datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Config / reference data
# ---------------------------------------------------------------------------

# Common brand names attackers impersonate — used for lookalike-domain checks.
# Extend this list with whatever brands are relevant to the org you're protecting.
WATCHED_BRANDS = [
    "microsoft", "office365", "google", "paypal", "apple", "amazon",
    "docusign", "dropbox", "chase", "wellsfargo", "bankofamerica",
    "linkedin", "adobe", "netflix", "zoom", "okta",
]

# Characters commonly used in homograph / lookalike attacks, mapped to the
# Latin character they're meant to imitate.
CONFUSABLES = {
    "0": "o", "1": "l", "1": "i", "3": "e", "5": "s", "@": "a",
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y",  # Cyrillic look-alikes
    "і": "i", "ѕ": "s", "ԁ": "d", "ɡ": "g",
}

URGENCY_KEYWORDS = [
    "urgent", "immediately", "action required", "verify your account",
    "suspended", "locked", "unauthorized", "expire", "final notice",
    "click here", "confirm your identity", "unusual activity",
]

FREEMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com",
    "icloud.com", "protonmail.com", "mail.com",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AuthResults:
    spf: Optional[str] = None          # pass / fail / softfail / neutral / none
    dkim: Optional[str] = None
    dmarc: Optional[str] = None

    @property
    def all_pass(self) -> bool:
        return self.spf == "pass" and self.dkim == "pass" and self.dmarc == "pass"

    @property
    def any_fail(self) -> bool:
        return "fail" in (self.spf, self.dkim, self.dmarc)


@dataclass
class DomainAnalysis:
    sender_domain: str = ""
    is_freemail: bool = False
    domain_age_days: Optional[int] = None
    domain_age_source: str = "unavailable"   # "unavailable" until you wire in real WHOIS
    lookalike_of: Optional[str] = None
    lookalike_score: float = 0.0
    reply_to_mismatch: bool = False
    display_name_mismatch: bool = False


@dataclass
class RiskAssessment:
    message_id: str = ""
    from_display: str = ""
    from_address: str = ""
    subject: str = ""
    auth: AuthResults = field(default_factory=AuthResults)
    domain: DomainAnalysis = field(default_factory=DomainAnalysis)
    urgency_hits: list = field(default_factory=list)
    link_domains: list = field(default_factory=list)
    link_domain_mismatch: bool = False
    infra_risk_score: float = 0.0     # 0-100
    verdict: str = "clean"            # clean | needs_review | escalate_to_layer2
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_auth_results(msg: Message) -> AuthResults:
    """Parse the Authentication-Results header for spf/dkim/dmarc verdicts."""
    header = msg.get("Authentication-Results", "") or ""
    # Some MTAs split this across multiple headers
    all_headers = msg.get_all("Authentication-Results", []) or []
    combined = header + " " + " ".join(all_headers)

    def extract(mechanism: str) -> Optional[str]:
        match = re.search(rf"{mechanism}=(\w+)", combined, re.IGNORECASE)
        return match.group(1).lower() if match else None

    return AuthResults(
        spf=extract("spf"),
        dkim=extract("dkim"),
        dmarc=extract("dmarc"),
    )


def _normalize_confusables(s: str) -> str:
    return "".join(CONFUSABLES.get(ch, ch) for ch in s.lower())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            )
        prev = cur
    return prev[-1]


def _check_lookalike(domain: str) -> tuple[Optional[str], float]:
    """
    Compare the sender domain's leading label against watched brand names.
    Two detection strategies, since attackers use both patterns:

    1. Whole-label similarity (edit distance) — catches 'micr0soft.com',
       'paypa1.com' where the whole label is a near-miss of the brand.
    2. Substring-with-confusables — catches 'micros0ft-support.com',
       'apple-id-verify.net' where the brand is embedded with extra
       words/hyphens tacked on, which breaks pure edit-distance.

    Returns (matched_brand, similarity_score 0-1).
    """
    root = domain.split(".")[0] if "." in domain else domain
    normalized = _normalize_confusables(root)

    best_brand, best_score = None, 0.0
    for brand in WATCHED_BRANDS:
        if brand == root:
            continue  # exact legitimate match, not a lookalike

        # Strategy 1: whole-label edit distance (short domains, typos)
        dist = min(_levenshtein(root, brand), _levenshtein(normalized, brand))
        max_len = max(len(brand), len(root))
        whole_label_score = 1 - (dist / max_len) if max_len else 0
        if dist <= 2 and dist > 0 and whole_label_score > best_score:
            best_brand, best_score = brand, whole_label_score

        # Strategy 2: brand embedded as substring plus extra tokens/hyphens
        # e.g. "micros0ft-support" contains normalized "microsoft"
        if brand in normalized and normalized != brand:
            # Score by how much of the label is "extra" beyond the brand name
            extra_chars = len(normalized) - len(brand)
            embedded_score = max(0.75, 1 - (extra_chars / (len(brand) * 3)))
            if embedded_score > best_score:
                best_brand, best_score = brand, embedded_score

    return best_brand, round(best_score, 2)


def _extract_domain(address: str) -> str:
    addr = parseaddr(address)[1]
    return addr.split("@")[-1].lower() if "@" in addr else ""


def _extract_link_domains(msg: Message) -> list[str]:
    """Pull hrefs / bare URLs out of the body (text + html parts)."""
    urls = set()
    url_pattern = re.compile(r'https?://([^\s/"\'<>]+)', re.IGNORECASE)

    def scan(payload: str):
        for m in url_pattern.finditer(payload):
            urls.add(m.group(1).lower())

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype in ("text/plain", "text/html"):
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        scan(payload.decode(errors="ignore"))
                except Exception:
                    continue
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                scan(payload.decode(errors="ignore"))
            else:
                scan(str(msg.get_payload()))
        except Exception:
            pass

    return sorted(urls)


def get_domain_age_days(domain: str) -> tuple[Optional[int], str]:
    """
    STUB: In production, wire this to a real WHOIS/RDAP client or a threat-intel
    API (e.g. WhoisXML, RDAP.org) — this environment has no network access.

    Contract: return (age_in_days, source_label). Return (None, "unavailable")
    if the lookup fails or is not configured, and the scorer will simply skip
    that signal rather than penalize on missing data.
    """
    return None, "unavailable"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_email(raw: str | bytes) -> RiskAssessment:
    msg = message_from_bytes(raw) if isinstance(raw, bytes) else message_from_string(raw)

    result = RiskAssessment()
    result.message_id = msg.get("Message-ID", "").strip()
    result.subject = msg.get("Subject", "")

    from_header = msg.get("From", "")
    display_name, from_address = parseaddr(from_header)
    result.from_display = display_name
    result.from_address = from_address

    result.auth = _parse_auth_results(msg)

    sender_domain = _extract_domain(from_address)
    result.domain.sender_domain = sender_domain
    result.domain.is_freemail = sender_domain in FREEMAIL_DOMAINS

    lookalike_brand, lookalike_score = _check_lookalike(sender_domain)
    result.domain.lookalike_of = lookalike_brand
    result.domain.lookalike_score = lookalike_score

    age_days, age_source = get_domain_age_days(sender_domain)
    result.domain.domain_age_days = age_days
    result.domain.domain_age_source = age_source

    # Reply-To mismatch: classic BEC / phishing tell
    reply_to = msg.get("Reply-To", "")
    if reply_to:
        reply_domain = _extract_domain(reply_to)
        if reply_domain and reply_domain != sender_domain:
            result.domain.reply_to_mismatch = True

    # Display name impersonation: e.g. "Microsoft Support <random@gmail.com>"
    if display_name:
        dn_lower = display_name.lower()
        for brand in WATCHED_BRANDS:
            if brand in dn_lower and brand not in sender_domain:
                result.domain.display_name_mismatch = True
                break

    # Urgency / social-engineering language in subject
    subject_lower = result.subject.lower()
    result.urgency_hits = [kw for kw in URGENCY_KEYWORDS if kw in subject_lower]

    # Link domain vs sender domain mismatch
    link_domains = _extract_link_domains(msg)
    result.link_domains = link_domains
    if link_domains and sender_domain:
        result.link_domain_mismatch = not any(
            sender_domain in ld or ld in sender_domain for ld in link_domains
        )

    result.infra_risk_score, result.reasons = _score(result)
    result.verdict = _verdict(result.infra_risk_score)

    return result


def _score(r: RiskAssessment) -> tuple[float, list[str]]:
    """Composite Infrastructure Risk Score (0-100), additive weighted signals."""
    score = 0.0
    reasons = []

    if r.auth.dmarc == "fail":
        score += 30; reasons.append("DMARC fail")
    elif r.auth.dmarc == "none":
        score += 8; reasons.append("No DMARC policy")

    if r.auth.spf == "fail":
        score += 20; reasons.append("SPF fail")
    if r.auth.dkim == "fail":
        score += 15; reasons.append("DKIM fail")

    if r.domain.lookalike_of and r.domain.lookalike_score >= 0.8:
        score += 25
        reasons.append(f"Lookalike domain of '{r.domain.lookalike_of}' "
                        f"(similarity {r.domain.lookalike_score})")

    if r.domain.display_name_mismatch:
        score += 20
        reasons.append("Display name impersonates a known brand not matching sender domain")

    if r.domain.reply_to_mismatch:
        score += 12
        reasons.append("Reply-To domain differs from From domain")

    if r.link_domain_mismatch:
        score += 15
        reasons.append("Embedded links point to a domain different from the sender")

    if r.urgency_hits:
        score += min(10, 3 * len(r.urgency_hits))
        reasons.append(f"Urgency/coercion language in subject: {r.urgency_hits}")

    if r.domain.domain_age_days is not None and r.domain.domain_age_days < 30:
        score += 15
        reasons.append(f"Sender domain registered {r.domain.domain_age_days} days ago")

    return min(round(score, 1), 100.0), reasons


def _verdict(score: float) -> str:
    if score >= 40:
        return "escalate_to_layer2"
    if score >= 15:
        return "needs_review"
    return "clean"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PhishShield Layer 1 — deterministic email risk filter")
    parser.add_argument("eml_path", help="Path to a .eml file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    with open(args.eml_path, "rb") as f:
        raw = f.read()

    result = analyze_email(raw)
    indent = 2 if args.pretty else None
    print(json.dumps(result.to_dict(), indent=indent, default=str))


if __name__ == "__main__":
    main()