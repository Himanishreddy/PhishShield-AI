"""
PhishShield AI — Nazario mbox Converter

Reads Nazario phishing corpus .mbox archives (each holding thousands of
messages) and writes a labelled CSV compatible with dataset_builder_v2.py.

Output schema — identical to convert_legit_emails.py so the two files merge
cleanly:

    text,label,source,category

    label    = 1 (phishing)
    source   = nazario_<year>   (per-file, so you can hold out a whole year
                                 as a cross-source test with --holdout-source)
    category = coarse attack type, metadata only, never a detection rule

Why per-year sources: keeping `nazario_2025` distinct from `nazario_2022`
lets you train on earlier years and evaluate on a later one, which is a
genuinely harder and more honest generalisation test than a random split
(phishing tactics drift over time).

PII note: these are attacker-authored emails, so there is no victim PII to
protect in the body text. We still redact recipient addresses that appear
in headers-turned-text, because those were real targets.

Usage:
    python convert_nazario.py
    python convert_nazario.py --in data_collection/nazario --out data_collection/nazario_phishing.csv
    python convert_nazario.py --max-per-file 2000     # cap for a faster first pass
"""

from __future__ import annotations

import argparse
import csv
import mailbox
import re
import sys
from collections import Counter
from email.header import decode_header, make_header
from email.message import Message
from pathlib import Path


# ---------------------------------------------------------------------------
# Extraction (mirrors convert_legit_emails.py so both corpora are processed
# identically — important, or the model could learn processing artefacts
# instead of real phishing signal)
# ---------------------------------------------------------------------------

def decode_subject(msg: Message) -> str:
    raw = msg.get("Subject", "") or ""
    try:
        return str(make_header(decode_header(raw))).strip()
    except Exception:
        return str(raw).strip()


def _decode_payload(part: Message) -> str:
    try:
        payload = part.get_payload(decode=True)
    except Exception:
        return ""
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError, AttributeError):
        try:
            return payload.decode("utf-8", errors="replace")
        except Exception:
            return ""


def strip_html(html: str) -> str:
    import html as html_mod
    text = re.sub(r"<(script|style|head)[^>]*>.*?</\1>", " ",
                  html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<\s*(br|/p|/div|/tr|/li|/h[1-6])\s*/?>", "\n",
                  text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_mod.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_body(msg: Message) -> str:
    plain, html = [], []
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.is_multipart():
                    continue
                disp = str(part.get("Content-Disposition", "") or "")
                if "attachment" in disp.lower():
                    continue
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    t = _decode_payload(part)
                    if t.strip():
                        plain.append(t)
                elif ctype == "text/html":
                    t = _decode_payload(part)
                    if t.strip():
                        html.append(t)
        else:
            t = _decode_payload(msg)
            (html if msg.get_content_type() == "text/html" else plain).append(t)
    except Exception:
        return ""

    if plain:
        return "\n".join(plain).strip()
    if html:
        return strip_html("\n".join(html))
    return ""


# ---------------------------------------------------------------------------
# Light redaction — victim addresses only
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
LONG_ID_RE = re.compile(r"(?<!\w)\d{8,}(?!\w)")


def redact(text: str) -> str:
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = LONG_ID_RE.sub("<ID>", text)
    return text


# ---------------------------------------------------------------------------
# Attack-type categorisation — METADATA ONLY.
# Used for per-category evaluation (e.g. "how well do we catch BEC vs
# credential phishing?"). Never consulted by the detection pipeline.
# ---------------------------------------------------------------------------

ATTACK_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("credential_phishing", (
        "password", "sign in", "signin", "log in", "login", "verify your account",
        "account verification", "credentials", "two-factor", "2fa",
        "authentication", "unusual activity", "suspended", "reset your password",
        "confirm your identity", "security alert",
    )),
    ("financial_fraud", (
        "wire transfer", "bank account", "payment", "invoice", "billing",
        "refund", "transaction", "credit card", "paypal", "swift",
        "beneficiary", "funds",
    )),
    ("bec_impersonation", (
        "urgent request", "are you available", "quick task", "gift card",
        "ceo", "cfo", "on behalf of", "confidential matter", "wire the",
    )),
    ("delivery_scam", (
        "package", "parcel", "shipment", "delivery", "tracking number",
        "customs", "courier", "dhl", "fedex", "ups ",
    )),
    ("brand_impersonation", (
        "microsoft", "office 365", "outlook", "google", "gmail", "apple",
        "icloud", "amazon", "netflix", "docusign", "dropbox", "linkedin",
    )),
    ("advance_fee", (
        "lottery", "inheritance", "beneficiary of", "next of kin",
        "million dollars", "business proposal", "barrister", "widow",
    )),
]


def categorise(subject: str, body: str) -> str:
    blob = f"{subject}\n{body}".lower()
    scores = {name: sum(1 for h in hints if h in blob)
              for name, hints in ATTACK_HINTS}
    scores = {k: v for k, v in scores.items() if v}
    return max(scores, key=scores.get) if scores else "phishing_general"


def combine(subject: str, body: str) -> str:
    subject, body = subject.strip(), body.strip()
    if subject and body:
        return f"{subject}\n\n{body}"
    return subject or body


def main():
    ap = argparse.ArgumentParser(description="Convert Nazario .mbox archives to labelled CSV")
    ap.add_argument("--in", dest="indir", default="data_collection/nazario",
                    help="Folder containing .mbox files")
    ap.add_argument("--out", default="data_collection/nazario_phishing.csv")
    ap.add_argument("--min-chars", type=int, default=30,
                    help="Skip messages whose extracted text is shorter than this")
    ap.add_argument("--max-chars", type=int, default=20000,
                    help="Truncate very long messages (keeps CSV manageable)")
    ap.add_argument("--max-per-file", type=int, default=None,
                    help="Cap messages taken from each mbox (useful for a quick first pass)")
    args = ap.parse_args()

    indir = Path(args.indir)
    if not indir.is_dir():
        sys.exit(f"Folder not found: {indir.resolve()}")

    # Accept .mbox files and extension-less Nazario downloads alike
    files = sorted([p for p in indir.iterdir()
                    if p.is_file() and p.suffix.lower() in (".mbox", "")
                    and p.name.lower() != "license.txt"])
    if not files:
        sys.exit(f"No mbox files found in {indir.resolve()}")

    print(f"Found {len(files)} mbox file(s):")
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    print()

    rows = []
    per_file_stats = {}
    total_failed = 0

    for path in files:
        # Derive a per-year source label: "phishing-2024.mbox" -> nazario_2024
        m = re.search(r"(19|20)\d{2}", path.stem)
        source = f"nazario_{m.group(0)}" if m else f"nazario_{path.stem}"

        kept = failed = 0
        try:
            mbox = mailbox.mbox(str(path))
        except Exception as e:
            print(f"  ! could not open {path.name}: {e}")
            continue

        for msg in mbox:
            if args.max_per_file and kept >= args.max_per_file:
                break
            try:
                subject = decode_subject(msg)
                body = extract_body(msg)
                text = combine(subject, body)
                if len(text.strip()) < args.min_chars:
                    failed += 1
                    continue
                text = redact(text)[:args.max_chars]
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                rows.append({
                    "text": text,
                    "label": 1,                      # phishing
                    "source": source,
                    "category": categorise(subject, body),
                })
                kept += 1
            except Exception:
                failed += 1

        per_file_stats[source] = {"kept": kept, "failed": failed}
        total_failed += failed
        print(f"  {source:18s} kept {kept:5d}   skipped {failed:4d}")

    # ---- Write ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "source", "category"])
        w.writeheader()
        w.writerows(rows)

    # ---- Report ----
    print("\n" + "=" * 60)
    print("CONVERSION REPORT")
    print("=" * 60)
    print(f"Total phishing emails: {len(rows)}")
    print(f"Skipped (empty/broken): {total_failed}")

    print("\nBy source year:")
    for src, st in per_file_stats.items():
        print(f"  {src:18s} {st['kept']:6d}")

    print("\nBy attack category:")
    for cat, n in Counter(r["category"] for r in rows).most_common():
        print(f"  {cat:24s} {n:6d}  ({n/max(len(rows),1):5.1%})")

    if rows:
        lengths = sorted(len(r["text"]) for r in rows)
        print(f"\nText length: min={lengths[0]} median={lengths[len(lengths)//2]} "
              f"max={lengths[-1]} chars")

    print(f"\nWrote {len(rows)} rows to {out_path.resolve()}")

    if rows:
        print("\n" + "=" * 60)
        print("PREVIEW (first 2)")
        print("=" * 60)
        for r in rows[:2]:
            prev = r["text"][:260].replace("\n", " / ")
            print(f"\nsource: {r['source']}  category: {r['category']}  label: {r['label']}")
            print(f"text  : {prev}...")

    print("\n" + "-" * 60)
    print("NEXT: feed this to dataset_builder_v2.py alongside your ham CSVs.")
    print("TIP: hold out one year as a cross-source test, e.g.")
    print("     --holdout-source nazario_2025")
    print("     Training on older years and testing on the newest is a much")
    print("     harder (and more honest) generalisation test than a random split.")


if __name__ == "__main__":
    main()