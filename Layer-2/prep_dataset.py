"""
PhishShield AI — Dataset Prep / Normalizer

Public phishing datasets each use their own column names and label values.
This script normalizes ANY of them into the clean `text,label` CSV format
that dataset_builder.py expects.

Workflow:
    1. Download a dataset CSV (from Hugging Face / Kaggle).
    2. Run this WITHOUT column args first to inspect its structure:
           python prep_dataset.py --in raw.csv --inspect
    3. Then run it WITH the right columns + label mapping to normalize:
           python prep_dataset.py --in raw.csv --out clean.csv \
               --text-col "Email Text" --label-col "Email Type" \
               --map "Safe Email=0,Phishing Email=1"

The output CSV has exactly two columns: text, label — ready to feed to
dataset_builder.py via --csv, or to drop into a labeled subfolder.

Examples for the recommended datasets:

  zefang-liu/phishing-email-dataset  (human phishing + ham -> labels 0/1)
      --text-col "Email Text" --label-col "Email Type"
      --map "Safe Email=0,Phishing Email=1"

  Kaggle LLM-generated set (kuladeep19)  (use phishing rows as ai_phish=2)
      --text-col body --label-col label
      --map "0=0,1=2"      # their 1 (phishing) becomes our 2 (ai_phish)
      # or keep legit rows as ham=0 as shown

Adjust the exact column names to whatever --inspect shows you.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def inspect(df: pd.DataFrame):
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        sample_vals = df[col].dropna().unique()[:5]
        # Truncate long text samples for readability
        preview = [str(v)[:60] + ("…" if len(str(v)) > 60 else "") for v in sample_vals]
        print(f"  {col!r}: {n_unique} unique values")
        print(f"      e.g. {preview}")
    print("\nFirst row in full:")
    if len(df):
        for col in df.columns:
            print(f"  {col}: {str(df.iloc[0][col])[:200]}")


def parse_label_map(s: str) -> dict:
    """Parse 'Safe Email=0,Phishing Email=1' into {'Safe Email': 0, ...}."""
    mapping = {}
    for pair in s.split(","):
        if "=" not in pair:
            raise ValueError(f"Bad --map entry {pair!r}; expected form KEY=VALUE")
        key, val = pair.rsplit("=", 1)
        mapping[key.strip()] = int(val.strip())
    return mapping


def main():
    ap = argparse.ArgumentParser(description="Normalize a phishing dataset into text,label CSV")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    ap.add_argument("--out", help="Output CSV path (required unless --inspect)")
    ap.add_argument("--inspect", action="store_true",
                    help="Just print the columns and sample values, then exit")
    ap.add_argument("--text-col", help="Column holding the email text/body")
    ap.add_argument("--label-col", help="Column holding the label")
    ap.add_argument("--map", dest="label_map",
                    help="Label value mapping, e.g. 'Safe Email=0,Phishing Email=1'. "
                         "Targets: 0=ham, 1=phishing, 2=ai_phish")
    ap.add_argument("--subject-col", default=None,
                    help="Optional subject column to prepend to the text")
    ap.add_argument("--drop-unmapped", action="store_true",
                    help="Drop rows whose label isn't in --map (instead of erroring)")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        sys.exit(f"Input file not found: {inp}")

    # Read robustly — some of these CSVs have quoting/encoding quirks
    try:
        df = pd.read_csv(inp, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(inp, encoding="latin-1")

    if args.inspect:
        inspect(df)
        return

    # Validate required args for normalization
    missing = [a for a in ("out", "text_col", "label_col")
               if not getattr(args, a.replace("-", "_"), None)]
    if missing:
        ap.error(f"--{', --'.join(m.replace('_','-') for m in missing)} required "
                 f"(or use --inspect). Run with --inspect first to see columns.")

    if args.text_col not in df.columns:
        sys.exit(f"--text-col {args.text_col!r} not found. Columns: {list(df.columns)}")
    if args.label_col not in df.columns:
        sys.exit(f"--label-col {args.label_col!r} not found. Columns: {list(df.columns)}")

    # Build text field (optionally subject + body)
    if args.subject_col and args.subject_col in df.columns:
        text = (df[args.subject_col].fillna("").astype(str) + "\n\n"
                + df[args.text_col].fillna("").astype(str)).str.strip()
    else:
        text = df[args.text_col].fillna("").astype(str).str.strip()

    out_df = pd.DataFrame({"text": text, "label": df[args.label_col]})

    # Apply label mapping
    if args.label_map:
        mapping = parse_label_map(args.label_map)
        # try matching as-is, then as strings (CSV labels are often strings)
        mapped = out_df["label"].map(mapping)
        if mapped.isna().any():
            mapped = out_df["label"].astype(str).str.strip().map(
                {str(k): v for k, v in mapping.items()}
            )
        unmapped_mask = mapped.isna()
        if unmapped_mask.any():
            bad_values = out_df.loc[unmapped_mask, "label"].astype(str).unique()[:10]
            if args.drop_unmapped:
                print(f"Dropping {unmapped_mask.sum()} rows with unmapped labels: {list(bad_values)}")
                out_df = out_df[~unmapped_mask]
                mapped = mapped[~unmapped_mask]
            else:
                sys.exit(f"These label values aren't in your --map: {list(bad_values)}\n"
                         f"Add them to --map or use --drop-unmapped.")
        out_df["label"] = mapped.astype(int)
    else:
        # assume label is already 0/1/2
        out_df["label"] = out_df["label"].astype(int)

    # Clean up: drop empties and duplicates
    before = len(out_df)
    out_df = out_df[out_df["text"].str.len() > 0]
    out_df = out_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {len(out_df)} rows to {out_path} (dropped {before - len(out_df)} empty/dupes)")
    print("Label distribution:")
    counts = out_df["label"].value_counts().sort_index()
    names = {0: "ham", 1: "phishing", 2: "ai_phish"}
    for lbl, n in counts.items():
        print(f"  {lbl} ({names.get(lbl, '?')}): {n}")


if __name__ == "__main__":
    main()