"""
PhishShield AI — SpamAssassin Corpus Converter

Reads the SpamAssassin public corpus (extracted folders of individual
message files, no file extensions) and writes a labelled CSV compatible
with dataset_builder_v2.py.

Output schema — identical to the other converters so all sources merge:

    text,label,source,category

WHY HAM ONLY BY DEFAULT
    SpamAssassin's `spam` folders are SPAM, not PHISHING. Spam is unwanted
    marketing; phishing is credential/financial theft. They overlap but are
    not the same class, and labelling bulk marketing as `phishing` would
    teach the model a boundary you do not want — "commercial language =
    phishing" is the same category of error as the "security vocabulary =
    phishing" bug we are trying to fix. Phishing examples come from Nazario
    instead. Pass --include-spam only if you have decided you want that
    class, and know it will be labelled 1.

WHY hard_ham MATTERS MOST
    `hard_ham` is legitimate mail that superficially resembles spam:
    commercial newsletters, notifications, promotional-but-real messages.
    These are exactly the hard negatives that teach "looks suspicious" !=
    "is malicious". `easy_ham` is ordinary correspondence and mainly adds
    general ham volume.

Folder -> label mapping:
    easy_ham, easy_ham_2, hard_ham   -> label 0 (legitimate)
    spam, spam_2                     -> label 1, ONLY with --include-spam

Usage:
    python convert_spamassassin.py
    python convert_spamassassin.py --in data_collection/spamassassin/extracted
    python convert_spamassassin.py --max-per-folder 1000
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from email import message_from_bytes
from email.header import decode_header, make_header
from email.message import Message
from pathlib import Path


# ---------------------------------------------------------------------------
# Extraction — identical logic to the other converters.
# This matters: if ham and phishing were cleaned differently, the model could
# learn the processing artefact instead of real signal, scoring well while
# generalising terribly.
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


EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
LONG_ID_RE = re.compile(r"(?<!\w)\d{8,}(?!\w)")


def redact(text: str) -> str:
    text = EMAIL_RE.sub("<EMAIL>", text)
    return LONG_ID_RE.sub("<ID>", text)


# ---------------------------------------------------------------------------
# Categorisation — metadata only, never a detection rule
# ---------------------------------------------------------------------------

CATEGORY_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("security_account", (
        "password", "sign in", "log in", "login", "account", "verify",
        "authentication", "security", "credentials", "subscription",
    )),
    ("commercial_newsletter", (
        "unsubscribe", "newsletter", "special offer", "discount", "sale",
        "promotion", "deal", "limited time", "click here to view",
    )),
    ("technical_list", (
        "mailing list", "listinfo", "patch", "bug", "kernel", "release",
        "commit", "repository", "developer", "sourceforge",
    )),
    ("transactional", (
        "order", "invoice", "receipt", "shipment", "confirmation",
        "payment", "billing", "statement",
    )),
]


def categorise(subject: str, body: str) -> str:
    blob = f"{subject}\n{body}".lower()
    scores = {name: sum(1 for h in hints if h in blob)
              for name, hints in CATEGORY_HINTS}
    scores = {k: v for k, v in scores.items() if v}
    return max(scores, key=scores.get) if scores else "general_correspondence"


def combine(subject: str, body: str) -> str:
    subject, body = subject.strip(), body.strip()
    if subject and body:
        return f"{subject}\n\n{body}"
    return subject or body


# Folder name -> (label, source suffix)
HAM_FOLDERS = {"easy_ham", "easy_ham_2", "hard_ham"}
SPAM_FOLDERS = {"spam", "spam_2"}


def main():
    ap = argparse.ArgumentParser(
        description="Convert the SpamAssassin corpus to labelled CSV (ham only by default)")
    ap.add_argument("--in", dest="indir",
                    default="data_collection/spamassassin/extracted",
                    help="Folder containing easy_ham/ hard_ham/ etc.")
    ap.add_argument("--out", default="data_collection/spamassassin_ham.csv")
    ap.add_argument("--include-spam", action="store_true",
                    help="Also convert spam/ folders as label 1. Off by default: "
                         "spam is not phishing, and conflating them teaches the "
                         "wrong boundary.")
    ap.add_argument("--max-per-folder", type=int, default=None,
                    help="Cap messages taken from each folder")
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--max-chars", type=int, default=20000)
    args = ap.parse_args()

    indir = Path(args.indir)
    if not indir.is_dir():
        sys.exit(f"Folder not found: {indir.resolve()}\n"
                 f"Extract the .tar files first, e.g.\n"
                 f"  tar -xf 20030228_hard_ham.tar -C data_collection/spamassassin/extracted")

    wanted = set(HAM_FOLDERS)
    if args.include_spam:
        wanted |= SPAM_FOLDERS

    # Find the message folders (they may be nested one level deep)
    folders = []
    for p in indir.rglob("*"):
        if p.is_dir() and p.name in wanted:
            folders.append(p)
    if not folders:
        sys.exit(f"No ham folders found under {indir.resolve()}\n"
                 f"Expected directories named: {sorted(wanted)}")

    print(f"Found {len(folders)} folder(s):")
    for f in sorted(folders):
        n = sum(1 for x in f.iterdir() if x.is_file())
        print(f"  {f.name:14s} {n:5d} files")
    print()

    rows = []
    stats = {}
    total_failed = 0

    for folder in sorted(folders):
        is_spam = folder.name in SPAM_FOLDERS
        label = 1 if is_spam else 0
        source = f"spamassassin_{folder.name}"
        kept = failed = 0

        files = sorted(x for x in folder.iterdir() if x.is_file())
        for path in files:
            if args.max_per_folder and kept >= args.max_per_folder:
                break
            # SpamAssassin includes a cmds file listing the messages — skip it
            if path.name.lower() in ("cmds",):
                continue
            try:
                msg = message_from_bytes(path.read_bytes())
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
                    "label": label,
                    "source": source,
                    "category": categorise(subject, body),
                })
                kept += 1
            except Exception:
                failed += 1

        stats[source] = {"kept": kept, "failed": failed, "label": label}
        total_failed += failed
        print(f"  {source:28s} label={label}  kept {kept:5d}  skipped {failed:4d}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "source", "category"])
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 60)
    print("CONVERSION REPORT")
    print("=" * 60)
    print(f"Total messages: {len(rows)}")
    print(f"Skipped:        {total_failed}")

    label_counts = Counter(r["label"] for r in rows)
    print(f"\nBy label:  ham(0)={label_counts.get(0,0)}  "
          f"spam(1)={label_counts.get(1,0)}")

    print("\nBy source:")
    for src, st in stats.items():
        print(f"  {src:28s} {st['kept']:6d}")

    print("\nBy category:")
    for cat, n in Counter(r["category"] for r in rows).most_common():
        print(f"  {cat:26s} {n:6d}  ({n/max(len(rows),1):5.1%})")

    if rows:
        lengths = sorted(len(r["text"]) for r in rows)
        print(f"\nText length: min={lengths[0]} median={lengths[len(lengths)//2]} "
              f"max={lengths[-1]}")

    print(f"\nWrote {len(rows)} rows to {out_path.resolve()}")

    if not args.include_spam:
        print("\nNOTE: spam/ folders were NOT converted. Spam is not phishing —")
        print("labelling marketing mail as phishing would teach the wrong")
        print("boundary. Phishing examples come from Nazario instead.")


if __name__ == "__main__":
    main()