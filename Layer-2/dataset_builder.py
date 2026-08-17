"""
PhishShield AI — Dataset Builder (Layer 2 foundation)

Assembles a labeled email corpus and produces clean train/val/test splits
ready for fine-tuning DistilBERT.

Three source classes:
    label 0 = ham       (legitimate email)
    label 1 = phishing  (human-written classic phishing)
    label 2 = ai_phish  (AI/LLM-generated phishing)

We keep AI-phishing as its own class so you can either:
    - train binary (ham vs. everything) by collapsing labels 1 & 2, or
    - train 3-class to show the model distinguishing human vs. AI phishing
      (this is your project's distinctive angle).

WHERE TO GET THE DATA (run these downloads on your own machine — this
sandbox has no network):

    Ham (legitimate):
        Enron email dataset — https://www.cs.cmu.edu/~enron/
        (or the cleaned CSV on Kaggle: "enron-email-dataset")

    Classic phishing:
        Nazario Phishing Corpus — https://monkey.org/~jose/phishing/
        SpamAssassin public corpus — https://spamassassin.apache.org/old/publiccorpus/

    AI-generated phishing (for the class-2 examples):
        Several public research datasets exist on Hugging Face —
        search "AI generated phishing" / "LLM phishing" datasets.
        Use an established, citable one so your results are reproducible.

Point --raw-dir at a folder of .eml files (any depth) OR pass CSVs via
--csv, and this script normalizes everything into one labeled dataset.

Usage:
    # From a folder of .eml files organized in ham/ phishing/ ai_phish/ subdirs:
    python dataset_builder.py --raw-dir ./raw_emails --out ./dataset

    # From CSVs (must have 'text' and 'label' columns):
    python dataset_builder.py --csv ham.csv --csv phishing.csv --out ./dataset

    # Validate the pipeline on the built-in sample (no data needed):
    python dataset_builder.py --demo --out ./dataset
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from email import message_from_bytes
from email.utils import parseaddr
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_NAMES = {0: "ham", 1: "phishing", 2: "ai_phish"}
LABEL_IDS = {v: k for k, v in LABEL_NAMES.items()}


# ---------------------------------------------------------------------------
# Text extraction (shared logic with Layer 1's parser)
# ---------------------------------------------------------------------------

def extract_text_from_eml(raw: bytes) -> tuple[str, str]:
    """Return (subject, body_text) from a raw .eml. Strips HTML crudely."""
    msg = message_from_bytes(raw)
    subject = msg.get("Subject", "") or ""

    body_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode(errors="ignore"))
            elif part.get_content_type() == "text/html" and not body_parts:
                payload = part.get_payload(decode=True)
                if payload:
                    body_parts.append(_strip_html(payload.decode(errors="ignore")))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            text = payload.decode(errors="ignore")
            if msg.get_content_type() == "text/html":
                text = _strip_html(text)
            body_parts.append(text)

    return subject.strip(), "\n".join(body_parts).strip()


def _strip_html(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def combine_subject_body(subject: str, body: str) -> str:
    """DistilBERT sees subject + body as one text field, subject first."""
    subject = (subject or "").strip()
    body = (body or "").strip()
    if subject and body:
        return f"{subject}\n\n{body}"
    return subject or body


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_from_raw_dir(raw_dir: Path) -> pd.DataFrame:
    """
    Expects subfolders named after labels, e.g.:
        raw_dir/ham/*.eml
        raw_dir/phishing/*.eml
        raw_dir/ai_phish/*.eml
    Any .eml at any depth under a label folder is picked up.
    """
    rows = []
    for label_name, label_id in LABEL_IDS.items():
        label_dir = raw_dir / label_name
        if not label_dir.exists():
            continue
        eml_files = list(label_dir.rglob("*.eml"))
        print(f"  {label_name:10s}: found {len(eml_files)} .eml files")
        for path in eml_files:
            try:
                raw = path.read_bytes()
                subject, body = extract_text_from_eml(raw)
                text = combine_subject_body(subject, body)
                if text.strip():
                    rows.append({"text": text, "label": label_id, "source": str(path.name)})
            except Exception as e:
                print(f"    ! skipped {path.name}: {e}")
    return pd.DataFrame(rows)


def load_from_csvs(csv_paths: list[str]) -> pd.DataFrame:
    """Each CSV must have 'text' and 'label' columns. label may be int or name."""
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{p} must have 'text' and 'label' columns; got {list(df.columns)}")
        # normalize label names -> ids
        if df["label"].dtype == object:
            df["label"] = df["label"].map(lambda x: LABEL_IDS.get(str(x).lower(), x))
        df["source"] = os.path.basename(p)
        frames.append(df[["text", "label", "source"]])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Built-in demo sample (validates the pipeline with zero external data)
# ---------------------------------------------------------------------------

def demo_dataframe() -> pd.DataFrame:
    """
    A tiny hand-written sample so you can run the whole pipeline end-to-end
    before you've downloaded any corpora. NOT for real training — far too
    small — just to prove the plumbing works.
    """
    samples = [
        # ham
        ("Notes from today's standup\n\nHey team, quick recap of what we covered. "
         "Docs are in the shared drive, ping me with questions.", 0),
        ("Lunch tomorrow?\n\nAre we still on for lunch at the usual place around noon? "
         "Let me know if the time still works.", 0),
        ("Q3 report draft attached\n\nHi all, attaching the first draft of the quarterly "
         "report for review. Comments welcome by Friday.", 0),
        ("Re: project timeline\n\nThanks for the update. The revised schedule looks good "
         "to me. Let's confirm at the sync on Monday.", 0),
        # classic human phishing (typo-ridden, crude)
        ("YOU HAVE WON!!!\n\nDear winner you have won $1,000,000 in our lottery. "
         "Send your bank details and processing fee of $200 to claim now!!!", 1),
        ("account verify needed\n\nYour acount has been limited. Click the link and "
         "confirm you informations or it will be closed in 24hrs.", 1),
        ("URGENT wire transfer\n\nkindly process payment urgent to below account, "
         "i am in meeting cannot talk, send confirmation when done.", 1),
        # AI-generated phishing (fluent, polished, structured)
        ("Action Required: Verify Your Account Security\n\nDear Valued Customer,\n\n"
         "We detected an unusual sign-in attempt on your account. To ensure your "
         "security, please verify your identity within 24 hours by visiting the "
         "secure portal below. We appreciate your prompt attention to this matter.", 2),
        ("Important Update Regarding Your Payroll Information\n\nHello,\n\nOur records "
         "indicate that your direct deposit details require confirmation following a "
         "recent system migration. Kindly review and confirm your information at your "
         "earliest convenience to avoid any disruption to your next payment.", 2),
        ("Your Invoice Is Ready for Review\n\nDear Colleague,\n\nPlease find attached "
         "the invoice pending your approval. To maintain compliance with our updated "
         "procurement process, we kindly ask that you review and authorize payment "
         "through the linked portal by end of business today.", 2),
    ]
    return pd.DataFrame(
        [{"text": t, "label": l, "source": "demo"} for t, l in samples]
    )


# ---------------------------------------------------------------------------
# Split + save
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42


def make_splits(df: pd.DataFrame, cfg: SplitConfig):
    """Stratified train/val/test so class balance is preserved in each split.

    Falls back to non-stratified splitting automatically when the dataset is
    too small for every class to appear in each split (e.g. the --demo sample).
    Real corpora with hundreds+ per class always stratify.
    """
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    n_classes = df["label"].nunique()
    test_n = int(len(df) * cfg.test_size)
    # Stratify only when the test split is big enough to hold one of each class
    can_stratify = n_classes > 1 and test_n >= n_classes

    if not can_stratify and n_classes > 1:
        print(f"  (dataset too small to stratify {n_classes} classes into a "
              f"{cfg.test_size:.0%} test split — using random split for this demo)")

    stratify = df["label"] if can_stratify else None
    train_val, test = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.seed, stratify=stratify
    )
    tv_classes = train_val["label"].nunique()
    tv_val_n = int(len(train_val) * (cfg.val_size / (1 - cfg.test_size)))
    tv_can_stratify = can_stratify and tv_classes > 1 and tv_val_n >= tv_classes
    stratify_tv = train_val["label"] if tv_can_stratify else None
    val_ratio = cfg.val_size / (1 - cfg.test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=cfg.seed, stratify=stratify_tv
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def summarize(name: str, df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts().sort_index().to_dict()
    named = {LABEL_NAMES.get(k, k): int(v) for k, v in counts.items()}
    print(f"  {name:6s} n={len(df):5d}  {named}")
    return named


def main():
    ap = argparse.ArgumentParser(description="PhishShield dataset builder")
    ap.add_argument("--raw-dir", type=str, help="Folder with ham/ phishing/ ai_phish/ subdirs of .eml files")
    ap.add_argument("--csv", action="append", default=[], help="CSV with text,label columns (repeatable)")
    ap.add_argument("--demo", action="store_true", help="Use built-in sample data (no downloads needed)")
    ap.add_argument("--out", type=str, default="./dataset", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading data...")
    frames = []
    if args.demo:
        frames.append(demo_dataframe())
    if args.raw_dir:
        frames.append(load_from_raw_dir(Path(args.raw_dir)))
    if args.csv:
        frames.append(load_from_csvs(args.csv))

    if not frames:
        ap.error("Provide at least one of --demo, --raw-dir, or --csv")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"\nTotal usable examples: {len(df)}")
    summarize("all", df)

    cfg = SplitConfig(seed=args.seed)
    train, val, test = make_splits(df, cfg)

    print("\nSplit sizes:")
    stats = {
        "train": summarize("train", train),
        "val": summarize("val", val),
        "test": summarize("test", test),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    with open(out / "label_map.json", "w") as f:
        json.dump(LABEL_NAMES, f, indent=2)
    with open(out / "stats.json", "w") as f:
        json.dump({"total": len(df), "splits": stats}, f, indent=2)

    print(f"\nWrote train/val/test CSVs + label_map.json to {out.resolve()}")
    if args.demo:
        print("\nNOTE: --demo data is only 10 examples — enough to test the pipeline, "
              "nowhere near enough to train a real model. Swap in the public corpora next.")


if __name__ == "__main__":
    main()