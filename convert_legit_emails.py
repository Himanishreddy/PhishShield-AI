"""
PhishShield AI — Legitimate Email Converter

Reads real .eml files exported from an inbox, extracts subject + body,
strips personally identifying information, assigns a coarse category, and
writes a CSV that dataset_builder_v2.py can consume directly.

Output schema (compatible with the Layer-2 dataset builder, which requires
`text` and `label` and preserves any extra columns through to the splits):

    text,label,source,category

    label    = 0 for every row (these are all legitimate emails)
    source   = real_inbox
    category = security_account | institutional | recruitment | academic
               | general_professional | uncategorised

IMPORTANT — what this script is and isn't:
    * It is a DATA PREPARATION tool. It does not detect anything, and the
      running detection pipeline never reads this CSV. These emails become
      training examples so the model can learn the legitimate/phishing
      distinction itself.
    * The `category` field is metadata for per-category evaluation only.
      It is NOT a detection rule and is never consulted at inference time.
    * These emails are for TRAINING. Do not reuse them as the OOD test set —
      measuring on data you trained on defeats the purpose.

PII handling:
    High-confidence patterns (email addresses, phone numbers, long digit
    runs, URL query tokens) are replaced with stable placeholders. Personal
    names are only stripped in the positions where they reliably appear
    (salutation and sign-off lines), because a broad "capitalised word"
    heuristic would also destroy meaningful words like Registrar, Monday or
    Finance — which would change the semantics the model needs to learn.

Usage:
    python convert_legit_emails.py
    python convert_legit_emails.py --in data_collection/legit --out data_collection/legit_ham.csv
    python convert_legit_emails.py --show-pii     # preview what got redacted
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from email import message_from_bytes
from email.header import decode_header, make_header
from email.message import Message
from pathlib import Path


# ---------------------------------------------------------------------------
# MIME parsing
# ---------------------------------------------------------------------------

def decode_subject(msg: Message) -> str:
    raw = msg.get("Subject", "") or ""
    try:
        return str(make_header(decode_header(raw))).strip()
    except Exception:
        return raw.strip()


def _decode_payload(part: Message) -> str:
    """Decode a single part's payload, honouring its declared charset."""
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("utf-8", errors="replace")


def strip_html(html: str) -> str:
    """Convert HTML to readable text without changing wording.

    Block-level tags become newlines so sentences don't run together;
    everything else is dropped and entities are unescaped.
    """
    import html as html_mod

    # Remove non-content elements entirely (including their contents)
    text = re.sub(r"<(script|style|head)[^>]*>.*?</\1>", " ",
                  html, flags=re.DOTALL | re.IGNORECASE)
    # Preserve structure: block tags -> newline
    text = re.sub(r"<\s*(br|/p|/div|/tr|/li|/h[1-6])\s*/?>", "\n",
                  text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_mod.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_body(msg: Message) -> str:
    """Extract the body, preferring text/plain and falling back to cleaned HTML.

    Walks all parts and collects the best available representation, skipping
    attachments (anything with a Content-Disposition of 'attachment').
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue
            disposition = str(part.get("Content-Disposition", "") or "")
            if "attachment" in disposition.lower():
                continue
            ctype = part.get_content_type()
            if ctype == "text/plain":
                t = _decode_payload(part)
                if t.strip():
                    plain_parts.append(t)
            elif ctype == "text/html":
                t = _decode_payload(part)
                if t.strip():
                    html_parts.append(t)
    else:
        text = _decode_payload(msg)
        if msg.get_content_type() == "text/html":
            html_parts.append(text)
        else:
            plain_parts.append(text)

    if plain_parts:
        return "\n".join(plain_parts).strip()
    if html_parts:
        return strip_html("\n".join(html_parts))
    return ""


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------

# Ordered: more specific patterns first so they win over general ones.
PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("<EMAIL>", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # URLs carrying tokens/session ids — keep the domain, drop the payload
    ("<URL>", re.compile(r"https?://\S*[?&](?:token|key|session|auth|id|code)=\S+",
                         re.IGNORECASE)),
    # International and local phone formats
    ("<PHONE>", re.compile(
        r"(?<!\w)(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,5}[\s-]?\d{3,5}(?:[\s-]?\d{2,5})?(?!\w)")),
    # Long digit runs: student/employee ids, account numbers, reference nos
    ("<ID>", re.compile(r"(?<!\w)\d{6,}(?!\w)")),
    # Mixed alphanumeric identifiers. Covers both orders seen in practice:
    #   EMP-4471, ABC/12345   (letters first)
    #   21CS4471, 2021BT0123  (digits first — common for student roll numbers)
    # Requires at least 2 letters and 3 digits so ordinary words and short
    # tokens like "COVID19" or "Q3" are left alone.
    ("<ID>", re.compile(r"(?<!\w)(?=[A-Za-z0-9/-]*[A-Za-z]{2})(?=[A-Za-z0-9/-]*\d{3})"
                        r"\d{0,4}[A-Za-z]{2,4}[-/]?\d{3,}(?!\w)")),
]

# Salutation / sign-off lines are the positions where personal names reliably
# appear. Restricting name redaction to these avoids damaging ordinary words.
SALUTATION_RE = re.compile(
    r"^(\s*(?:dear|hi|hello|hey|respected)\s+)([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,2})",
    re.IGNORECASE | re.MULTILINE)
SIGNOFF_RE = re.compile(
    r"^(\s*(?:regards|best regards|warm regards|sincerely|thanks|thank you|"
    r"yours (?:sincerely|faithfully|truly)|cheers|best)\s*,?\s*\n+\s*)"
    r"([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,2})\s*$",
    re.IGNORECASE | re.MULTILINE)

# Titles that are roles, not personal names — keep them, they carry meaning
ROLE_WORDS = {
    "sir", "madam", "all", "team", "student", "students", "colleagues",
    "everyone", "staff", "faculty", "member", "members", "customer",
    "user", "candidate", "applicant", "participants", "registrar",
    "principal", "director", "coordinator", "admin", "administrator",
}


def redact_pii(text: str, track: list[str] | None = None) -> str:
    """Replace high-confidence PII with placeholders.

    Semantics are preserved: only identifiers are replaced, never the
    surrounding wording, structure, or any content-bearing vocabulary.
    """
    out = text

    for placeholder, pattern in PII_PATTERNS:
        def _sub(m):
            if track is not None:
                track.append(f"{placeholder}: {m.group(0)[:60]}")
            return placeholder
        out = pattern.sub(_sub, out)

    def _name_sub(m):
        prefix, name = m.group(1), m.group(2)
        # Don't redact role words ("Dear All", "Dear Sir", "Dear Students")
        if name.strip().split()[0].lower() in ROLE_WORDS:
            return m.group(0)
        if track is not None:
            track.append(f"<NAME>: {name}")
        return f"{prefix}<NAME>"

    out = SALUTATION_RE.sub(_name_sub, out)
    out = SIGNOFF_RE.sub(lambda m: f"{m.group(1)}<NAME>"
                         if m.group(2).strip().split()[0].lower() not in ROLE_WORDS
                         else m.group(0), out)
    return out


# ---------------------------------------------------------------------------
# Categorisation (metadata only — never a detection rule)
# ---------------------------------------------------------------------------

# These keyword sets label emails for PER-CATEGORY EVALUATION so we can see
# which registers the model struggles with. They are written to a metadata
# column and are NEVER used by the detection pipeline at inference time.
CATEGORY_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("security_account", (
        "password", "sign-in", "signin", "log in", "login", "two-factor",
        "2fa", "authentication", "verify your", "verification code",
        "security alert", "unusual activity", "new device", "account access",
        "mfa", "credentials", "session expired", "reset your",
    )),
    ("recruitment", (
        "interview", "job application", "resume", "cv", "vacancy",
        "hiring", "recruit", "shortlist", "offer letter", "candidate",
        "placement", "internship",
    )),
    ("academic", (
        "semester", "examination", "exam ", "syllabus", "lecture",
        "assignment", "coursework", "grade", "transcript", "faculty",
        "attendance", "curriculum", "thesis", "convocation",
    )),
    ("institutional", (
        "circular", "notice", "hereby", "by the direction", "all concerned",
        "registrar", "principal", "administration", "hostel", "campus",
        "notification", "office order", "is informed",
    )),
]


def categorise(subject: str, body: str) -> str:
    """Assign a coarse category from keyword hints. Metadata only."""
    blob = f"{subject}\n{body}".lower()
    scores = {}
    for name, hints in CATEGORY_HINTS:
        hits = sum(1 for h in hints if h in blob)
        if hits:
            scores[name] = hits
    if not scores:
        return "general_professional"
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def combine(subject: str, body: str) -> str:
    """Subject first, then body — matching how the Layer-2 model was trained."""
    subject, body = subject.strip(), body.strip()
    if subject and body:
        return f"{subject}\n\n{body}"
    return subject or body


def main():
    ap = argparse.ArgumentParser(description="Convert legitimate .eml files to a labeled CSV")
    ap.add_argument("--in", dest="indir", default="data_collection/legit",
                    help="Folder containing .eml files")
    ap.add_argument("--out", default="data_collection/legit_ham.csv",
                    help="Output CSV path")
    ap.add_argument("--source", default="real_inbox", help="Value for the `source` column")
    ap.add_argument("--min-chars", type=int, default=20,
                    help="Skip emails whose extracted text is shorter than this")
    ap.add_argument("--show-pii", action="store_true",
                    help="Print what was redacted (review before sharing the CSV)")
    args = ap.parse_args()

    indir = Path(args.indir)
    if not indir.is_dir():
        sys.exit(f"Input folder not found: {indir.resolve()}\n"
                 f"Put your .eml files there, or pass --in <folder>.")

    eml_files = sorted(indir.rglob("*.eml"))
    if not eml_files:
        sys.exit(f"No .eml files found in {indir.resolve()}")

    print(f"Found {len(eml_files)} .eml file(s) in {indir}\n")

    rows, failures = [], []
    pii_log: list[str] = []

    for path in eml_files:
        try:
            raw = path.read_bytes()
            msg = message_from_bytes(raw)

            # Python's parser accepts almost anything, treating a plain text
            # file as a header-less body. Require at least one recognisable
            # email header so genuinely malformed files are reported rather
            # than silently converted into a training example.
            has_headers = any(msg.get(h) for h in
                              ("Subject", "From", "To", "Date", "Message-ID"))
            if not has_headers:
                failures.append((path.name, "no email headers found — not a valid .eml"))
                continue

            subject = decode_subject(msg)
            body = extract_body(msg)
            text = combine(subject, body)

            if len(text.strip()) < args.min_chars:
                failures.append((path.name, "no usable text extracted"))
                continue

            tracked: list[str] = [] if args.show_pii else None
            clean = redact_pii(text, track=tracked)
            if tracked:
                pii_log.extend(f"  [{path.name}] {t}" for t in tracked)

            # Collapse excessive blank lines without altering wording
            clean = re.sub(r"\n{3,}", "\n\n", clean).strip()

            rows.append({
                "text": clean,
                "label": 0,                       # all legitimate
                "source": args.source,
                "category": categorise(subject, body),
            })
        except Exception as e:
            failures.append((path.name, f"{type(e).__name__}: {e}"))

    # ---- Write ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source", "category"])
        writer.writeheader()
        writer.writerows(rows)

    # ---- Report ----
    print("=" * 60)
    print("CONVERSION REPORT")
    print("=" * 60)
    print(f"Processed successfully: {len(rows)}")
    print(f"Failed:                 {len(failures)}")
    if failures:
        print("\nFailures:")
        for name, reason in failures:
            print(f"  {name}: {reason}")

    if rows:
        from collections import Counter
        print("\nCategory breakdown:")
        for cat, n in Counter(r["category"] for r in rows).most_common():
            print(f"  {cat:24s} {n}")

        lengths = [len(r["text"]) for r in rows]
        print(f"\nText length: min={min(lengths)} median="
              f"{sorted(lengths)[len(lengths)//2]} max={max(lengths)} chars")

    if args.show_pii:
        print(f"\nRedactions ({len(pii_log)}):")
        for line in pii_log[:60]:
            print(line)
        if len(pii_log) > 60:
            print(f"  ... and {len(pii_log) - 60} more")

    print(f"\nWrote {len(rows)} rows to {out_path.resolve()}")

    # ---- Preview ----
    if rows:
        print("\n" + "=" * 60)
        print("PREVIEW (first 3 rows)")
        print("=" * 60)
        for r in rows[:3]:
            preview = r["text"][:300].replace("\n", " ⏎ ")
            print(f"\ncategory : {r['category']}")
            print(f"label    : {r['label']}   source: {r['source']}")
            print(f"text     : {preview}{'...' if len(r['text']) > 300 else ''}")

    print("\n" + "-" * 60)
    print("NEXT: review the CSV (especially redactions) before using it.")
    print("These are TRAINING examples — keep them separate from your OOD")
    print("test set, or you'll be measuring on data you trained on.")


if __name__ == "__main__":
    main()